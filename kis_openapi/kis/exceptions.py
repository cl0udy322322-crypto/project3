class KISError(Exception):
    """Base exception for KIS client."""


class KISAuthError(KISError):
    """Authentication/authorization failures."""


class KISRequestError(KISError):
    """HTTP request failures (non-2xx or parse errors)."""
