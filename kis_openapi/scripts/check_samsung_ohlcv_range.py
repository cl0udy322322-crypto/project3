#!/usr/bin/env python3
"""Check earliest/latest OHLCV available for Samsung (005930) via KIS"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

# Ensure workspace root is in sys.path so package imports work when run as script
PKG_DIR = Path(__file__).resolve().parents[1]  # .../kis_openapi
WS_DIR = Path(__file__).resolve().parents[2]  # workspace root
if str(WS_DIR) not in sys.path:
    sys.path.insert(0, str(WS_DIR))

# Load env helper if present (prefer .env inside kis_openapi)
try:
    from kis_openapi.kis.utils import load_dotenv_if_present
    load_dotenv_if_present(PKG_DIR / ".env")
except Exception:
    pass

try:
    from kis_openapi.kis import KISClient, KISConfig
    # Load Backtest's data_loader by path to avoid package import issues
    import importlib.util
    DATA_LOADER_PATH = Path(WS_DIR) / "Backtest" / "backtest" / "data_loader.py"
    # Use the module name that matches package layout to satisfy dataclass resolution
    spec = importlib.util.spec_from_file_location("backtest.data_loader", str(DATA_LOADER_PATH))
    data_loader = importlib.util.module_from_spec(spec)
    # Insert into sys.modules under the expected name to avoid dataclass annotation resolution issues
    sys.modules[spec.name] = data_loader
    spec.loader.exec_module(data_loader)
    DataLoader = data_loader.DataLoader
    DataConfig = data_loader.DataConfig
except Exception as e:
    print("Import error:", e)
    traceback.print_exc()
    sys.exit(1)

# Build client
cfg = KISConfig.from_env()
print("KIS base_url:", cfg.base_url)
print("Has app key:", bool(cfg.app_key))
# Try explicit token request to get more debug info if it fails
try:
    from kis_openapi.kis.auth import request_token
    try:
        tok = request_token(cfg.base_url, cfg.app_key, cfg.app_secret)
        print("Token acquired (preview):", tok.access_token[:8] + "...", "expires_at:", tok.expires_at)
    except Exception as e:
        print("Token request failed (exception):", e)
        import json
        import requests

        url = f"{cfg.base_url.rstrip('/')}/oauth2/tokenP"
        payload_obj = {"grant_type": "client_credentials", "appkey": cfg.app_key, "appsecret": cfg.app_secret}
        payload_str = json.dumps(payload_obj, ensure_ascii=False)

        def dump_resp(tag: str, r: requests.Response) -> None:
            print(f"[{tag}] status:", r.status_code)
            print(f"[{tag}] content-type:", r.headers.get("Content-Type"))
            print(f"[{tag}] content-length header:", r.headers.get("Content-Length"))
            print(f"[{tag}] len(content):", len(r.content or b""))
            print(f"[{tag}] headers (subset):", {k: v for k, v in r.headers.items() if k.lower() in {"content-type","content-length","transfer-encoding","server","date"}})
            print(f"[{tag}] content preview (bytes):", repr((r.content or b"")[:200]))
            txt = (r.content or b"").decode("utf-8", errors="replace")
            print(f"[{tag}] text preview (utf-8, replace):", repr(txt[:500]))

        try:
            r1 = requests.post(url, json=payload_obj, timeout=10)
            dump_resp("json=", r1)
        except Exception as e2:
            print("Token endpoint json= request error:", e2)

        try:
            r2 = requests.post(
                url,
                data=payload_str,
                headers={"Content-Type": "application/json", "Accept": "text/plain", "charset": "UTF-8"},
                timeout=10,
            )
            dump_resp("data=json.dumps", r2)
        except Exception as e2:
            print("Token endpoint data= request error:", e2)
        raise
except Exception:
    # If auth helper isn't available or fails, continue and let client later fail with full traceback
    pass

client = KISClient(cfg)
loader = DataLoader(kis_client=client, kis_cfg=cfg)

conf = DataConfig(
    source="kis",
    kis_config={
        "data_type": "stock",
        "stock_code": "005930",
        "days": 30000,  # large window to request maximum history
    },
)

try:
    df = loader.load_ohlcv(conf)
    print("ROWS:", len(df))
    if not df.empty:
        print("EARLIEST:", df.index.min())
        print("LATEST:", df.index.max())
        na_counts = df[["open", "high", "low", "close", "volume"]].isna().sum().to_dict()
        print("NA_COUNTS:", na_counts)
        print("SAMPLE_HEAD:\n", df.head().to_string())
        print("SAMPLE_TAIL:\n", df.tail().to_string())
    else:
        print("No data returned (empty DataFrame)")
except Exception as e:
    print("Exception during fetch:")
    traceback.print_exc()
    sys.exit(2)

print("Done")
