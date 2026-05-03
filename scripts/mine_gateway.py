from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from secret_refs import read_mine_config, resolve_secret_ref

DEFAULT_GATEWAY_BASE_URL = "http://127.0.0.1:18789/v1"
DEFAULT_GATEWAY_MODEL = "openclaw/default"


def resolve_mine_gateway_model_config() -> dict[str, Any]:
    if not _mine_gateway_enabled():
        return {}

    token = _env_value("MINE_GATEWAY_TOKEN", "OPENCLAW_GATEWAY_TOKEN")
    if not token:
        token = _read_gateway_token_from_config().strip()
    if not token:
        return {}

    config: dict[str, Any] = {
        "provider": _env_value("MINE_GATEWAY_PROVIDER", "OPENCLAW_GATEWAY_PROVIDER", default="openclaw"),
        "base_url": _env_value("MINE_GATEWAY_BASE_URL", "OPENCLAW_GATEWAY_BASE_URL", default=DEFAULT_GATEWAY_BASE_URL),
        "api_key": token,
        "model": _env_value("MINE_ENRICH_MODEL", "OPENCLAW_ENRICH_MODEL", default=DEFAULT_GATEWAY_MODEL),
    }
    upstream_model = _env_value("MINE_UPSTREAM_MODEL", "OPENCLAW_UPSTREAM_MODEL")
    if upstream_model:
        config["openclaw_model"] = upstream_model
    return config


def write_model_config(path: Path, model_config: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _mine_gateway_enabled() -> bool:
    mode = _env_value("MINE_ENRICH_MODE", "OPENCLAW_ENRICH_MODE", default="auto").lower()
    if mode in {"0", "false", "off", "disabled"}:
        return False
    # auto and gateway both enable gateway; in auto mode CLI wins, gateway is fallback
    return True


def _read_gateway_token_from_config() -> str:
    payload = read_mine_config()
    token = (((payload.get("gateway") or {}).get("auth") or {}).get("token"))
    if isinstance(token, str):
        return token
    return resolve_secret_ref(token, payload)


def _env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default
