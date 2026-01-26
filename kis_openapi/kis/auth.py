from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import requests

from .exceptions import KISAuthError, KISRequestError


@dataclass
class KISToken:
    access_token: str
    token_type: str
    expires_at: datetime

    def is_expired(self, skew_seconds: int = 30) -> bool:
        return datetime.now(timezone.utc) >= (self.expires_at - timedelta(seconds=skew_seconds))


def _parse_token_response(data: dict[str, Any]) -> KISToken:
    access_token = data.get("access_token")
    token_type = data.get("token_type", "Bearer")

    if not access_token:
        raise KISAuthError(f"Token response missing access_token: {data}")

    # KIS commonly returns expires_in (seconds)
    expires_in = data.get("expires_in")
    if isinstance(expires_in, (int, float)):
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
    else:
        # Fallback: unknown expiry -> short lifetime
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    return KISToken(access_token=str(access_token), token_type=str(token_type), expires_at=expires_at)


def load_cached_token(cache_path: str | None) -> KISToken | None:
    if not cache_path:
        return None
    p = Path(cache_path)
    if not p.exists():
        return None

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        expires_at = datetime.fromisoformat(raw["expires_at"]).astimezone(timezone.utc)
        token = KISToken(
            access_token=raw["access_token"],
            token_type=raw.get("token_type", "Bearer"),
            expires_at=expires_at,
        )
        if token.is_expired():
            return None
        return token
    except Exception:
        return None


def save_cached_token(cache_path: str | None, token: KISToken) -> None:
    if not cache_path:
        return
    p = Path(cache_path)
    payload = {
        "access_token": token.access_token,
        "token_type": token.token_type,
        "expires_at": token.expires_at.astimezone(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def request_token(base_url: str, app_key: str, app_secret: str, timeout: float = 10.0) -> KISToken:
    """Request OAuth2 client_credentials token.

    Endpoint is typically: POST /oauth2/tokenP
    """

    url = f"{base_url.rstrip('/')}/oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret,
    }

    try:
        resp = requests.post(url, json=body, timeout=timeout)
    except requests.RequestException as e:
        raise KISRequestError(f"Token request failed: {e}") from e

    if resp.status_code // 100 != 2:
        raise KISAuthError(f"Token request failed: {resp.status_code} {resp.text}")

    try:
        data = resp.json()
    except ValueError as e:
        raise KISAuthError(f"Token response not JSON: {resp.text}") from e

    return _parse_token_response(data)
