"""WebSocket client for receiving task assignments from the platform.

Used by both miner (repeat_crawl_task) and validator (evaluation_task).
Both share the same WS endpoint: /api/mining/v1/ws

Resilience features:
- Exponential backoff with jitter to avoid thundering herd on reconnect
- Application-level ping/pong keepalive to detect dead connections early
- Stale connection detection (no data received for too long triggers reconnect)
- Graceful handling of close frames vs unexpected disconnects
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from typing import Any, Callable

log = logging.getLogger("validator.ws")

# Keepalive: send a ping every 20s, expect pong within 10s
PING_INTERVAL = 20.0
PING_TIMEOUT = 10.0

# If no data (messages or pongs) received for this long, consider connection stale
STALE_CONNECTION_TIMEOUT = 90.0


class WSDisconnected(Exception):
    """Raised when the WebSocket connection is lost."""


class WSMessage:
    """Parsed WebSocket message from the platform."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.type: str = str(raw.get("type") or "")
        self.data: dict[str, Any] = raw.get("data") if isinstance(raw.get("data"), dict) else {}

    @property
    def task_id(self) -> str:
        return str(self.data.get("task_id") or "")

    @property
    def assignment_id(self) -> str:
        return str(self.data.get("assignment_id") or "")

    @property
    def submission_id(self) -> str:
        return str(self.data.get("submission_id") or "")

    @property
    def mode(self) -> str:
        return str(self.data.get("mode") or "single")

    @property
    def repeat_crawl_task_id(self) -> str:
        """Task ID from repeat_crawl_task message."""
        return str(self.data.get("id") or "")

    def __repr__(self) -> str:
        return f"WSMessage(type={self.type!r}, task_id={self.task_id!r}, assignment_id={self.assignment_id!r})"


class ValidatorWSClient:
    """
    Manages WebSocket connection to the platform for receiving tasks.

    Protocol:
      Server -> Client: {"type": "evaluation_task", "data": {"task_id": "evt_xxx"}}
      Server -> Client: {"type": "repeat_crawl_task", "data": {...full task...}}
      Server -> Client: {"type": "error", "code": "...", "message": "...", "retry_after_seconds": N}
      Client -> Server: {"ack": "<task_id>"}          (repeat crawl task ACK, triggers claim)
      Client -> Server: {"reject": "<task_id>"}       (reject repeat crawl task)

    Evaluation flow: WS notify (task_id only) → HTTP POST /evaluation-tasks/claim
    (gets assignment_id + full data) → evaluate → HTTP POST /evaluation-tasks/{id}/report

    Reconnection:
      Uses exponential backoff with jitter: base 1s -> 2s -> 4s -> ... -> 60s max,
      plus random jitter of ±25% to avoid thundering herd.
      On auth failure (401), refreshes wallet session before reconnecting.

    Keepalive:
      Sends WebSocket-level pings every 20s. If no pong is received within 10s,
      the connection is considered dead and a reconnect is triggered.
      Additionally, if no data at all is received for 90s, the connection is
      proactively recycled (some proxies/LBs silently drop idle connections).
    """

    def __init__(
        self,
        *,
        ws_url: str,
        auth_headers: dict[str, str],
        on_auth_refresh: Callable[[], dict[str, str]] | None = None,
        ping_interval: float = PING_INTERVAL,
        ping_timeout: float = PING_TIMEOUT,
        stale_timeout: float = STALE_CONNECTION_TIMEOUT,
    ) -> None:
        self._ws_url = ws_url
        self._auth_headers = auth_headers
        self._on_auth_refresh = on_auth_refresh
        self._ws: Any = None  # websockets connection object
        self._connected = False
        self._reconnect_attempt = 0
        self._max_backoff = 60
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._closed = False

        # Keepalive settings
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._stale_timeout = stale_timeout

        # Tracking timestamps for health monitoring
        self._last_data_received: float = 0.0  # any data (message or pong)
        self._last_ping_sent: float = 0.0
        self._awaiting_pong = False
        self._ping_thread: threading.Thread | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_data_received(self) -> float:
        """Timestamp of last received data (for external health monitoring)."""
        return self._last_data_received

    def connect(self) -> None:
        """Establish WebSocket connection with auth headers."""
        try:
            import websockets.sync.client as ws_sync
        except ImportError as exc:
            self._connected = False
            raise WSDisconnected(
                "websockets not installed — run: pip install websockets"
            ) from exc

        # Close any existing connection cleanly before reconnecting
        self._close_ws_quietly()

        try:
            extra_headers = dict(self._auth_headers)
            self._ws = ws_sync.connect(
                self._ws_url,
                additional_headers=extra_headers,
                open_timeout=15,
                close_timeout=5,
                # Enable library-level ping as a secondary safety net
                ping_interval=self._ping_interval,
                ping_timeout=self._ping_timeout,
            )
            self._connected = True
            self._reconnect_attempt = 0
            self._last_data_received = time.monotonic()
            self._awaiting_pong = False
            log.info("WebSocket connected to %s", self._ws_url)
            # Start keepalive monitor thread
            self._start_ping_thread()
        except Exception as exc:
            self._connected = False
            log.error("WebSocket connect failed: %s", exc)
            raise WSDisconnected(f"connect failed: {exc}") from exc

    def reopen(self) -> None:
        """Allow reconnections after close() — used when restarting the receive loop."""
        self._closed = False
        self._connected = False
        self._stop_event.clear()

    def close(self) -> None:
        """Close the WebSocket connection."""
        self._stop_event.set()
        self._stop_ping_thread()
        with self._lock:
            self._closed = True
            self._connected = False
            ws = self._ws
            self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def send_ack_eval(self, task_id: str) -> None:
        """Send evaluation task ACK (triggers claim). Must be within 30s."""
        self._send({"ack_eval": task_id})
        log.info("Sent ack_eval for task %s", task_id)

    def send_ack_repeat_crawl(self, task_id: str) -> None:
        """Acknowledge repeat crawl task, starts 5-min lease."""
        self._send({"ack": task_id})
        log.info("Sent ack for repeat crawl task %s", task_id)

    def send_reject_repeat_crawl(self, task_id: str) -> None:
        """Reject repeat crawl task, no penalty."""
        self._send({"reject": task_id})
        log.info("Sent reject for repeat crawl task %s", task_id)

    def receive(self, timeout: float = 30.0) -> WSMessage | None:
        """
        Receive next message from WebSocket.
        Returns None on timeout, raises WSDisconnected on connection loss.

        Also checks for stale connections — if no data has been received
        for longer than stale_timeout, proactively disconnects.
        """
        with self._lock:
            if not self._connected or self._ws is None:
                raise WSDisconnected("not connected")
            ws = self._ws

        # Proactive stale connection detection
        if self._last_data_received > 0:
            elapsed = time.monotonic() - self._last_data_received
            if elapsed > self._stale_timeout:
                log.warning(
                    "No data received for %.0fs (stale_timeout=%.0fs) — forcing reconnect",
                    elapsed,
                    self._stale_timeout,
                )
                self._connected = False
                self._close_ws_quietly()
                raise WSDisconnected("connection stale — no data received")

        try:
            raw = ws.recv(timeout=timeout)
            # Update last data timestamp on any successful receive
            self._last_data_received = time.monotonic()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                log.warning("Received non-dict message: %s", raw[:200])
                return None
            msg = WSMessage(data)
            log.debug("Received: %s", msg)
            return msg
        except TimeoutError:
            return None
        except json.JSONDecodeError as exc:
            # Still got data, just malformed — update timestamp but warn
            self._last_data_received = time.monotonic()
            log.warning("Invalid JSON from WebSocket: %s", exc)
            return None
        except Exception as exc:
            self._connected = False
            # Distinguish clean close from unexpected disconnect
            exc_str = str(exc).lower()
            if "close" in exc_str or "1000" in exc_str or "1001" in exc_str:
                log.info("WebSocket closed cleanly by server — will reconnect")
            else:
                log.warning("WebSocket receive error: %s", exc)
            raise WSDisconnected(f"receive failed: {exc}") from exc

    def reconnect_with_backoff(self) -> None:
        """Reconnect with exponential backoff + jitter. Refreshes auth if needed."""
        if self._closed:
            return

        self._reconnect_attempt += 1
        # Exponential backoff with ±25% jitter
        base_delay = min(2 ** max(self._reconnect_attempt - 1, 0), self._max_backoff)
        jitter = base_delay * 0.25 * (2 * random.random() - 1)  # ±25%
        delay = max(0.5, base_delay + jitter)

        log.info(
            "Reconnecting in %.1fs (attempt %d, base=%ds)...",
            delay,
            self._reconnect_attempt,
            base_delay,
        )
        if self._stop_event.wait(timeout=delay):
            return  # stop requested during backoff

        # Refresh auth headers if callback provided
        if self._on_auth_refresh is not None:
            try:
                self._auth_headers = self._on_auth_refresh()
                log.info("Auth headers refreshed for reconnection")
            except Exception as exc:
                log.warning("Auth refresh failed: %s", exc)

        try:
            self.connect()
        except WSDisconnected:
            log.warning("Reconnect attempt %d failed", self._reconnect_attempt)

    def is_healthy(self) -> bool:
        """Check if the connection appears healthy (for external monitoring)."""
        if not self._connected:
            return False
        if self._last_data_received <= 0:
            return True  # just connected, no data expected yet
        elapsed = time.monotonic() - self._last_data_received
        return elapsed < self._stale_timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, data: dict[str, Any]) -> None:
        with self._lock:
            if not self._connected or self._ws is None:
                raise WSDisconnected("not connected")
            ws = self._ws
        try:
            ws.send(json.dumps(data))
        except Exception as exc:
            with self._lock:
                self._connected = False
            raise WSDisconnected(f"send failed: {exc}") from exc

    def _close_ws_quietly(self) -> None:
        """Close the underlying WS object without changing client state flags."""
        with self._lock:
            ws = self._ws
            self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _start_ping_thread(self) -> None:
        """Start background thread that monitors connection health."""
        self._stop_ping_thread()
        t = threading.Thread(
            target=self._ping_loop, name="ws-keepalive", daemon=True
        )
        self._ping_thread = t
        t.start()

    def _stop_ping_thread(self) -> None:
        """Stop the keepalive monitor thread."""
        t = self._ping_thread
        self._ping_thread = None
        # Thread will exit on next iteration when _connected is False or _stop_event is set

    def _ping_loop(self) -> None:
        """
        Periodically check connection health.

        The websockets library handles protocol-level ping/pong automatically
        (we pass ping_interval/ping_timeout to connect()). This loop serves as
        an additional application-level monitor that detects stale connections
        where the library's ping might not catch issues (e.g., half-open TCP).
        """
        while self._connected and not self._stop_event.is_set():
            if self._stop_event.wait(timeout=self._ping_interval):
                break
            if not self._connected:
                break

            # Check if connection has gone stale
            if self._last_data_received > 0:
                elapsed = time.monotonic() - self._last_data_received
                if elapsed > self._stale_timeout:
                    log.warning(
                        "Keepalive monitor: no data for %.0fs — marking disconnected",
                        elapsed,
                    )
                    self._connected = False
                    self._close_ws_quietly()
                    break

            # Try sending an application-level ping as a JSON message
            # (some servers respond to this; if send fails, connection is dead)
            try:
                with self._lock:
                    ws = self._ws
                if ws is not None:
                    ws.ping()
                    log.debug("Keepalive ping sent")
            except Exception as exc:
                log.warning("Keepalive ping failed: %s — marking disconnected", exc)
                self._connected = False
                break

        log.debug("Keepalive monitor thread exited")
