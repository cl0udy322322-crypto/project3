from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class KISConfig:
    base_url: str
    app_key: str
    app_secret: str

    # Optional account info
    cano: str | None = None
    acnt_prdt_cd: str | None = None

    # Default customer type used by some endpoints: P(개인) / B(법인)
    custtype: str = "P"

    # Token cache path (optional)
    token_cache_path: str | None = ".kis_token_cache.json"

    @staticmethod
    def from_env(prefix: str = "KIS_") -> "KISConfig":
        def getenv(name: str, default: str | None = None) -> str | None:
            return os.getenv(f"{prefix}{name}", default)

        base_url = getenv("BASE_URL")
        app_key = getenv("APP_KEY")
        app_secret = getenv("APP_SECRET")

        if not base_url:
            raise ValueError(f"Missing env: {prefix}BASE_URL")
        if not app_key:
            raise ValueError(f"Missing env: {prefix}APP_KEY")
        if not app_secret:
            raise ValueError(f"Missing env: {prefix}APP_SECRET")

        cano = getenv("CANO") or None
        acnt_prdt_cd = getenv("ACNT_PRDT_CD") or None
        custtype = (getenv("CUSTTYPE", "P") or "P").strip() or "P"
        token_cache_path = getenv("TOKEN_CACHE", ".kis_token_cache.json")

        return KISConfig(
            base_url=base_url.rstrip("/"),
            app_key=app_key,
            app_secret=app_secret,
            cano=cano,
            acnt_prdt_cd=acnt_prdt_cd,
            custtype=custtype,
            token_cache_path=token_cache_path or None,
        )
