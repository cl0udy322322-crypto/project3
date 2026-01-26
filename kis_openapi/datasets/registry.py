from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from kis_openapi.kis import KISClient, KISConfig
from kis_openapi.kis.marketdata import fetch_index_daily


@dataclass(frozen=True)
class Dataset:
    name: str
    out_dir: Path
    out_file: str
    fetch: Callable[[KISClient, KISConfig, int], pd.DataFrame]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    description: str
    build: Callable[[Path, dict[str, str]], Dataset]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _build_index_daily(out_root: Path, params: dict[str, str]) -> Dataset:
    index_iscd = (params.get("iscd") or "2001").strip() or "2001"
    market_div = (params.get("mrkt") or "U").strip() or "U"
    tr_id = (params.get("tr_id") or "FHKUP03500100").strip() or "FHKUP03500100"

    out_dir = out_root / "indices" / "index" / index_iscd

    def _fetch(client: KISClient, cfg: KISConfig, days: int) -> pd.DataFrame:
        return fetch_index_daily(
            client,
            cfg,
            index_iscd=index_iscd,
            market_div=market_div,
            tr_id=tr_id,
            days=days,
        )

    return Dataset(
        name=f"index_daily[{index_iscd}]",
        out_dir=out_dir,
        out_file="daily.csv",
        fetch=_fetch,
    )


def _build_kospi200_daily(out_root: Path, params: dict[str, str]) -> Dataset:
    # Keep a stable key/path for the common case.
    out_dir = out_root / "indices" / "kospi200"

    def _fetch(client: KISClient, cfg: KISConfig, days: int) -> pd.DataFrame:
        return fetch_index_daily(
            client,
            cfg,
            index_iscd="2001",
            market_div="U",
            tr_id="FHKUP03500100",
            days=days,
        )

    return Dataset(
        name="kospi200_daily",
        out_dir=out_dir,
        out_file="daily.csv",
        fetch=_fetch,
    )


def specs() -> dict[str, DatasetSpec]:
    """Dataset specs registry.

    - This is code-only and safe to commit.
    - Actual outputs go under `data/` which should be gitignored.
    """

    return {
        "kospi200_daily": DatasetSpec(
            key="kospi200_daily",
            description="KOSPI200 일별 지수(기본: close) CSV 저장",
            build=_build_kospi200_daily,
        ),
        "index_daily": DatasetSpec(
            key="index_daily",
            description="국내 지수 일별(파라미터: iscd, mrkt=U, tr_id) CSV 저장",
            build=_build_index_daily,
        ),
        # VI dataset is intentionally left as a spec placeholder until the exact KIS endpoint/fields are decided.
        # "vi_events": DatasetSpec(...)
    }


def build_dataset(out_root: Path, key: str, params: dict[str, str] | None = None) -> Dataset:
    params = params or {}
    reg = specs()
    if key not in reg:
        keys = ", ".join(sorted(reg.keys()))
        raise KeyError(f"Unknown dataset key: {key}. Available: {keys}")
    return reg[key].build(out_root, params)


def run_dataset(ds: Dataset, client: KISClient, cfg: KISConfig, days: int) -> Path:
    _ensure_dir(ds.out_dir)
    df = ds.fetch(client, cfg, days)
    out_path = ds.out_dir / ds.out_file
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path
