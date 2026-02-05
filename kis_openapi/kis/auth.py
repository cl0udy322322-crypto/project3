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

    def _post_json() -> requests.Response:
        # KIS official samples often send a JSON string in the request body (data=) and
        # set Accept to text/plain. Some gateways behave differently depending on these.
        payload = json.dumps(body, ensure_ascii=False)
        return requests.post(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/plain",
                "charset": "UTF-8",
            },
            timeout=timeout,
        )

    def _post_form() -> requests.Response:
        # KIS docs typically show JSON; however, certain gateways accept/require form encoding.
        return requests.post(
            url,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )

    try:
        resp = _post_json()
    except requests.RequestException as e:
        raise KISRequestError(f"Token request failed: {e}") from e

    # If we got a non-2xx, try to surface response body to help debugging.
    if resp.status_code // 100 != 2:
        raise KISAuthError(
            f"Token request failed: {resp.status_code} "
            f"content_type={resp.headers.get('Content-Type')} body={resp.text}"
        )

    # Some users observe HTTP 200 with empty body. In that case, retry as form-encoded.
    if not resp.content:
        try:
            resp2 = _post_form()
            if resp2.status_code // 100 == 2 and resp2.content:
                resp = resp2
            elif resp2.status_code // 100 != 2:
                raise KISAuthError(
                    f"Token request confirm failed: {resp2.status_code} "
                    f"content_type={resp2.headers.get('Content-Type')} body={resp2.text}"
                )
        except requests.RequestException as e:
            raise KISRequestError(f"Token request retry failed: {e}") from e

    try:
        data = resp.json()
    except ValueError as e:
        # Provide richer context without leaking credentials.
        ct = resp.headers.get("Content-Type")
        body_preview = (resp.text or "")[:1000]
        raise KISAuthError(
            f"Token response not JSON: status={resp.status_code} content_type={ct} "
            f"len={len(resp.content)} body_preview={body_preview!r}"
        ) from e

    return _parse_token_response(data)
