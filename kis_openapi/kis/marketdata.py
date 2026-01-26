from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import os
import sys
from typing import Any, Iterable

import pandas as pd

from .client import KISClient
from .config import KISConfig


@dataclass(frozen=True)
class IndexDailyBar:
    dt: date
    close: float


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _extract_output_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("output2", "output1", "output"):
        val = payload.get(key)
        if isinstance(val, list):
            return [r for r in val if isinstance(r, dict)]
    return []


def _pick_first(d: dict[str, Any], keys: Iterable[str]) -> str | None:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _parse_index_bar(row: dict[str, Any]) -> IndexDailyBar | None:
    ds = _pick_first(
        row,
        [
            "stck_bsop_date",
            "bsop_date",
            "idx_bsop_date",
            "date",
        ],
    )
    cs = _pick_first(
        row,
        [
            "idx_clpr",
            "bstp_nmix_prpr",
            "stck_clpr",
            "close",
        ],
    )

    if not ds or not cs:
        return None

    try:
        dt = datetime.strptime(ds, "%Y%m%d").date()
    except ValueError:
        return None

    try:
        close = float(cs.replace(",", ""))
    except ValueError:
        return None

    return IndexDailyBar(dt=dt, close=close)


def fetch_index_daily(
    client: KISClient,
    cfg: KISConfig,
    *,
    index_iscd: str,
    market_div: str = "U",
    tr_id: str = "FHKUP03500100",
    days: int = 365,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Fetch domestic index daily close series via KIS.

    This is a reusable primitive. You can build dataset fetchers on top of it.

    Notes:
    - For this endpoint, `fid_cond_mrkt_div_code` is commonly `U` (not `J`).
    """

    end_dt = end or date.today()
    start_dt = start or (end_dt - timedelta(days=days))

    path = "/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice"

    params = {
        "fid_cond_mrkt_div_code": (market_div or "U").strip() or "U",
        "fid_input_iscd": (index_iscd or "").strip(),
        "fid_input_date_1": _yyyymmdd(start_dt),
        "fid_input_date_2": _yyyymmdd(end_dt),
        "fid_period_div_code": "D",
        "fid_org_adj_prc": "0",
    }
    if not params["fid_input_iscd"]:
        raise ValueError("index_iscd is required")

    headers = {
        "tr_id": (tr_id or "FHKUP03500100").strip() or "FHKUP03500100",
        "custtype": cfg.custtype,
    }

    payload = client.request("GET", path, params=params, headers=headers)

    rt_cd = str(payload.get("rt_cd", "")).strip()
    if rt_cd and rt_cd != "0":
        raise RuntimeError(
            f"KIS error rt_cd={rt_cd} msg_cd={payload.get('msg_cd')} msg1={payload.get('msg1')}"
        )

    rows = _extract_output_rows(payload)
    bars: list[IndexDailyBar] = []
    for r in rows:
        bar = _parse_index_bar(r)
        if bar is not None:
            bars.append(bar)

    if not bars:
        top_keys = sorted(payload.keys())
        sample_keys = sorted(set().union(*(r.keys() for r in rows[:3]))) if rows else []
        raise RuntimeError(
            "No parsable bars returned. "
            f"top_keys={top_keys} sample_keys={sample_keys} "
            f"(used: iscd={params['fid_input_iscd']} mrkt={params['fid_cond_mrkt_div_code']} tr_id={headers['tr_id']})"
        )

    df = (
        pd.DataFrame({"date": [b.dt for b in bars], "close": [b.close for b in bars]})
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    df = df[df["date"] >= start_dt]
    return df


def fetch_kospi200_daily(client: KISClient, cfg: KISConfig, days: int = 365) -> pd.DataFrame:
    """Fetch KOSPI200 daily close for recent N days via KIS.

    Convenience wrapper around `fetch_index_daily`.
    """

    # Allow overrides via env for quick adjustment.
    index_iscd = os.getenv("KIS_INDEX_ISCD", "2001").strip() or "2001"
    market_div = os.getenv("KIS_INDEX_MRKT_DIV", "U").strip() or "U"
    tr_id = os.getenv("KIS_INDEX_DAILY_TR_ID", "FHKUP03500100").strip() or "FHKUP03500100"

    return fetch_index_daily(
        client,
        cfg,
        index_iscd=index_iscd,
        market_div=market_div,
        tr_id=tr_id,
        days=days,
    )


def fetch_stock_price(
    client: KISClient,
    cfg: KISConfig,
    *,
    stock_code: str,
    tr_id: str | None = None,
) -> dict[str, Any]:
    """Fetch current stock price via KIS.

    Args:
        client: KISClient instance
        cfg: KISConfig instance
        stock_code: Stock code (e.g., "005930" for 삼성전자)
        tr_id: Transaction ID (default: FHKST01010100 for real, VTTC0802U for mock)

    Returns:
        Dictionary containing stock price information
    """
    # Determine TR_ID based on base_url (mock vs real)
    # 참고: 주식 현재가 조회 TR_ID는 환경에 따라 다를 수 있음
    if tr_id is None:
        is_vts = "openapivts" in cfg.base_url.lower()
        # 모의투자와 실전투자의 TR_ID가 다를 수 있음
        # 일부 환경에서는 다른 TR_ID를 사용해야 할 수 있음
        tr_id = "VTTC0802U" if is_vts else "FHKST01010100"

    path = "/uapi/domestic-stock/v1/quotations/inquire-price"

    # KIS API 파라미터 시도: 소문자와 대문자 모두 시도
    # 일반적으로 소문자로 시작하는 경우가 많음
    params = {
        "fid_cond_mrkt_div_code": "J",  # J: 주식
        "fid_input_iscd": stock_code.strip(),
    }

    if not params["fid_input_iscd"]:
        raise ValueError("stock_code is required")

    headers = {
        "tr_id": tr_id.strip(),
        "custtype": cfg.custtype,
    }

    payload = client.request("GET", path, params=params, headers=headers)

    rt_cd = str(payload.get("rt_cd", "")).strip()
    if rt_cd and rt_cd != "0":
        raise RuntimeError(
            f"KIS error rt_cd={rt_cd} msg_cd={payload.get('msg_cd')} msg1={payload.get('msg1')}"
        )

    output = payload.get("output", {})
    if not output:
        raise RuntimeError(f"No output returned. payload keys: {list(payload.keys())}")

    return output
