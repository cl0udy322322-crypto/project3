from __future__ import annotations

import os
from pathlib import Path
import sys

PKG_DIR = Path(__file__).resolve().parents[1]  # .../kis_openapi
WS_DIR = Path(__file__).resolve().parents[2]  # workspace root
if str(WS_DIR) not in sys.path:
    sys.path.insert(0, str(WS_DIR))

from kis_openapi.kis import KISClient, KISConfig
from kis_openapi.kis.utils import load_dotenv_if_present


def main() -> None:
    load_dotenv_if_present(PKG_DIR / ".env")
    cfg = KISConfig.from_env()
    client = KISClient(cfg)

    info = client.ping()
    print("KIS client ready")
    print(f"token expires_at: {info['expires_at']}")


if __name__ == "__main__":
    main()
