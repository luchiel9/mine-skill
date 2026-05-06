"""DNS-caching transport for httpx.

Resolves hostnames once, caches the result, and re-resolves on connection
failure or TTL expiry. Eliminates repeated DNS lookups per request while
handling IP rotation gracefully.

Usage:
    from dns_cache import DNSCacheTransport

    client = httpx.Client(
        base_url="https://api.minework.net",
        transport=DNSCacheTransport(),
    )
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any, Iterable

import httpcore
import httpx
from httpcore._backends.sync import SyncBackend, SyncStream
from httpx._transports.default import ResponseStream

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 300  # seconds


class DNSCache:
    """Thread-safe DNS cache with TTL and failure-triggered refresh."""

    def __init__(self, ttl: float = _DEFAULT_TTL) -> None:
        self._ttl = ttl
        self._cache: dict[str, tuple[str, float]] = {}  # host -> (ip, expires_at)
        self._lock = threading.Lock()

    def resolve(self, host: str, port: int = 443) -> str:
        """Return cached IP or resolve and cache."""
        with self._lock:
            entry = self._cache.get(host)
            if entry and entry[1] > time.monotonic():
                return entry[0]

        # Resolve outside lock to avoid blocking other threads
        ip = self._do_resolve(host, port)

        with self._lock:
            self._cache[host] = (ip, time.monotonic() + self._ttl)

        return ip

    def invalidate(self, host: str) -> None:
        """Force re-resolve on next access (call on connection failure)."""
        with self._lock:
            self._cache.pop(host, None)
        logger.info("DNS cache invalidated for %s", host)

    def _do_resolve(self, host: str, port: int) -> str:
        """Perform actual DNS resolution."""
        start = time.perf_counter()
        results = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        elapsed_ms = (time.perf_counter() - start) * 1000
        ip = results[0][4][0]
        logger.info("DNS resolved %s -> %s (%.1fms)", host, ip, elapsed_ms)
        return ip


# Global shared cache instance
_global_cache = DNSCache()


class CachedDNSBackend(SyncBackend):
    """Network backend that resolves hostnames via DNSCache.

    Intercepts connect_tcp calls, resolves the host through the cache,
    and connects to the cached IP. TLS SNI remains correct because httpcore
    handles SNI separately from the TCP connection address.
    """

    def __init__(self, dns_cache: DNSCache | None = None) -> None:
        super().__init__()
        self._dns_cache = dns_cache or _global_cache

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[tuple[int, int, int] | tuple[int, int, bytes | bytearray] | tuple[int, int, None, int]] | None = None,
    ) -> SyncStream:
        # Check if host is already an IP (skip resolution)
        try:
            socket.inet_aton(host)
            is_ip = True
        except OSError:
            is_ip = False

        if not is_ip:
            resolved_host = self._dns_cache.resolve(host, port)
        else:
            resolved_host = host

        return super().connect_tcp(
            resolved_host,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )


class DNSCacheTransport(httpx.BaseTransport):
    """httpx transport that uses cached DNS resolution via custom network backend.

    TLS works correctly because httpcore uses the original hostname for SNI
    negotiation, while the network backend connects to the cached IP.

    On connection errors, invalidates the DNS cache entry so next attempt
    re-resolves (handles IP rotation).
    """

    def __init__(
        self,
        dns_cache: DNSCache | None = None,
        verify: Any = True,
        cert: Any = None,
        http1: bool = True,
        http2: bool = False,
        limits: httpx.Limits = httpx.Limits(),
        trust_env: bool = True,
        retries: int = 0,
        socket_options: Any = None,
    ) -> None:
        self._dns_cache = dns_cache or _global_cache
        self._backend = CachedDNSBackend(dns_cache=self._dns_cache)

        import ssl as _ssl
        if verify is True:
            ssl_context = _ssl.create_default_context()
        elif verify is False:
            ssl_context = _ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = _ssl.CERT_NONE
        elif isinstance(verify, str):
            ssl_context = _ssl.create_default_context(cafile=verify)
        else:
            ssl_context = verify

        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl_context,
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            http1=http1,
            http2=http2,
            retries=retries,
            network_backend=self._backend,
            socket_options=socket_options,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host or ""

        # Build httpcore request
        req = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
        )

        try:
            resp = self._pool.handle_request(req)
        except httpcore.ConnectError as exc:
            self._dns_cache.invalidate(host)
            raise httpx.ConnectError(str(exc), request=request) from exc
        except httpcore.ConnectTimeout as exc:
            self._dns_cache.invalidate(host)
            raise httpx.ConnectTimeout(str(exc), request=request) from exc

        return httpx.Response(
            status_code=resp.status,
            headers=resp.headers,
            stream=ResponseStream(resp.stream),
            extensions=resp.extensions,
        )

    def close(self) -> None:
        self._pool.close()


def create_cached_client(
    base_url: str = "",
    ttl: float = _DEFAULT_TTL,
    dns_cache: DNSCache | None = None,
    **client_kwargs: Any,
) -> httpx.Client:
    """Create an httpx.Client with DNS caching enabled.

    Args:
        base_url: Base URL for the client.
        ttl: DNS cache TTL in seconds.
        dns_cache: Optional shared DNSCache instance. Uses global if None.
        **client_kwargs: Additional kwargs passed to httpx.Client.

    Returns:
        httpx.Client with DNS-caching transport.
    """
    cache = dns_cache or _global_cache
    if ttl != _DEFAULT_TTL and dns_cache is None:
        cache = DNSCache(ttl=ttl)

    transport = DNSCacheTransport(dns_cache=cache)
    return httpx.Client(base_url=base_url, transport=transport, **client_kwargs)
