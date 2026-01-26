from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PKG_DIR = Path(__file__).resolve().parents[1]  # .../kis_openapi
WS_DIR = Path(__file__).resolve().parents[2]  # workspace root
if str(WS_DIR) not in sys.path:
    sys.path.insert(0, str(WS_DIR))

from kis_openapi.kis import KISClient, KISConfig
from kis_openapi.kis.utils import load_dotenv_if_present
from kis_openapi.datasets import build_dataset, run_dataset, specs


def _parse_params(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"Invalid --param: {it} (expected key=value)")
        k, v = it.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise SystemExit(f"Invalid --param: {it} (empty key)")
        out[k] = v
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch datasets via KIS and store outputs under a data/ tree")
    p.add_argument("dataset", nargs="?", help="dataset key (e.g. kospi200_daily, index_daily)")
    p.add_argument("--days", type=int, default=365, help="lookback days (default: 365)")
    p.add_argument(
        "--list",
        action="store_true",
        help="list available datasets and exit",
    )
    p.add_argument(
        "--param",
        action="append",
        default=[],
        help="dataset parameter, repeatable (e.g. --param iscd=2001 --param mrkt=U)",
    )
    p.add_argument(
        "--out-root",
        default=None,
        help="output root directory (default: <pkg>/data)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        reg = specs()
        for k in sorted(reg.keys()):
            print(f"{k}: {reg[k].description}")
        return

    if not args.dataset:
        raise SystemExit("Missing dataset. Use --list to see available datasets.")

    load_dotenv_if_present(PKG_DIR / ".env")
    cfg = KISConfig.from_env()
    client = KISClient(cfg)

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else (PKG_DIR / "data")
    params = _parse_params(args.param)
    try:
        ds = build_dataset(out_root, args.dataset, params)
    except KeyError as e:
        raise SystemExit(f"{e}. Use --list to see available datasets.") from None
    out_path = run_dataset(ds, client, cfg, days=args.days)
    print(f"saved: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
