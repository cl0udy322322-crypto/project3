from __future__ import annotations

from typing import Any

import requests

from .auth import KISToken, load_cached_token, request_token, save_cached_token
from .config import KISConfig
from .exceptions import KISRequestError


class KISClient:
    """Small KIS OpenAPI client skeleton.

    - Manages OAuth token (client_credentials)
    - Provides a thin request wrapper
    - Leaves endpoint-specific implementation to user
    """

    def __init__(
        self,
        config: KISConfig,
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.config = config
        self._session = session or requests.Session()
        self._timeout = timeout
        self._token: KISToken | None = load_cached_token(config.token_cache_path)

    def get_access_token(self, force_refresh: bool = False) -> KISToken:
        if (not force_refresh) and self._token and (not self._token.is_expired()):
            return self._token

        token = request_token(
            base_url=self.config.base_url,
            app_key=self.config.app_key,
            app_secret=self.config.app_secret,
            timeout=self._timeout,
        )
        self._token = token
        save_cached_token(self.config.token_cache_path, token)
        return token

    def _default_headers(self) -> dict[str, str]:
        token = self.get_access_token()
        return {
            "authorization": f"{token.token_type} {token.access_token}",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        req_headers = self._default_headers()
        if headers:
            req_headers.update(headers)

        try:
            resp = self._session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_body,
                headers=req_headers,
                timeout=timeout or self._timeout,
            )
        except requests.RequestException as e:
            raise KISRequestError(f"Request failed: {e}") from e

        if resp.status_code // 100 != 2:
            raise KISRequestError(f"HTTP {resp.status_code}: {resp.text}")

        try:
            return resp.json()
        except ValueError as e:
            raise KISRequestError(f"Response not JSON: {resp.text}") from e

    # ---- Example endpoint stubs (fill as needed) ----

    def ping(self) -> dict[str, Any]:
        """Connectivity smoke test.

        Note: KIS doesn't provide a universal ping endpoint; this uses token issuance only.
        You can replace this with a lightweight GET endpoint you actually use.
        """
        token = self.get_access_token()
        return {
            "ok": True,
            "expires_at": token.expires_at.isoformat(),
        }
