from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from kis_openapi.kis import KISClient, KISConfig
from kis_openapi.kis.marketdata import fetch_kospi200_daily as fetch_kospi200_daily_md


@dataclass(frozen=True)
class Dataset:
    name: str
    out_dir: Path
    out_file: str
    fetch: Callable[[KISClient, KISConfig, int], pd.DataFrame]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def fetch_kospi200_daily_dataset(client: KISClient, cfg: KISConfig, days: int) -> pd.DataFrame:
    return fetch_kospi200_daily_md(client, cfg, days=days)


def registry(base_dir: Path) -> dict[str, Dataset]:
    data_dir = base_dir / "data"
    return {
        "kospi200_daily": Dataset(
            name="kospi200_daily",
            out_dir=data_dir / "indices" / "kospi200",
            out_file="daily.csv",
            fetch=fetch_kospi200_daily_dataset,
        ),
        # Placeholder for VI dataset(s)
        # "vi_events": Dataset(...)
    }


def run_dataset(ds: Dataset, client: KISClient, cfg: KISConfig, days: int) -> Path:
    _ensure_dir(ds.out_dir)
    df = ds.fetch(client, cfg, days)
    out_path = ds.out_dir / ds.out_file
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path
