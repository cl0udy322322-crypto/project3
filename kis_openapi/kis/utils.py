from __future__ import annotations

from pathlib import Path


def load_dotenv_if_present(env_path: str | Path = ".env") -> None:
    """Load .env if present (optional dependency: python-dotenv).

    Accepts a path so callers can keep a consistent directory layout
    without changing the process working directory.
    """

    env_path = Path(env_path)
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path)
    except Exception:
        # If python-dotenv isn't installed or any issue occurs, just skip.
        return
