#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
매크로(거시경제) 지표 1. USD/KRW vs KOSPI: 음(-)의 상관관계(디커플링)
---------------------------------------------------------------------

[운용사(퀀트/리스크) 관점]
- 환율(달러/원, USD/KRW)은 한국 주식에 대해 흔히 '리스크 프록시'로 작동합니다.
  * USD/KRW 상승(원화 약세) = 글로벌/국내 위험회피(Risk-Off) 경향 강화 → 주식에 부담
  * USD/KRW 하락(원화 강세) = 위험선호(Risk-On) 및 외국인 수급 환경 개선 가능 → 주식에 우호적
- 다만 이 관계는 **항상 일정하지 않으며**, 국면(성장/금리/신용/수급)에 따라
  상관이 약해지거나 반대로 움직이기도 합니다. 본 스크립트는 레짐 변화를 데이터로 확인합니다.

[무엇을 계산하나?]
1) USD/KRW, KOSPI 지수(종가) 수집 후 정렬/정합
2) 일간(또는 월간) 수익률로 변환
3) 전체 구간 상관 + 최근 구간 상관 요약
4) 롤링 상관(기본 60영업일)로 '디커플링(음의 상관)' 구간 시각화
5) 산점도(수익률)로 관계를 직관적으로 확인
6) (추가) 코스피 '국지적 고점'과 달러/원 '국지적 저점'이 같은 날(또는 근접한 기간)에 발생한 지점을 표시

[주의(리스크/해석)]
- 상관은 과거의 동행 관계이며, 미래를 보장하지 않습니다.
- 환율-주가 관계는 원인/결과가 섞여 있을 수 있습니다(수급/정책/리스크 이벤트).
- 이벤트 구간에서는 상관이 급격히 뒤틀릴 수 있습니다.
- 교육/연구용 예시이며 투자자문이 아닙니다.

[데이터 소스]
- FinanceDataReader: 'USD/KRW' 환율, 'KS11' 코스피 지수 심볼 사용

실행 예시
- 기본(일간, 60일 롤링):  python usdkrw_kospi_decoupling_kor.py
- 월간(12개월 롤링):      python usdkrw_kospi_decoupling_kor.py --freq M --window 12
- 그림 저장:              python usdkrw_kospi_decoupling_kor.py --save --outdir outputs
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import FinanceDataReader as fdr


# -----------------------------
# Font: Korean-friendly
# -----------------------------
def setup_korean_font() -> None:
    """
    Try to set a Korean-capable font on Windows/macOS/Linux.
    If none is available, matplotlib will fall back; labels may break.
    """
    candidates = [
        "Malgun Gothic",   # Windows
        "AppleGothic",     # macOS
        "NanumGothic",     # common on Linux / user-installed
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    available = {f.name for f in plt.matplotlib.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False  # show minus sign properly


# -----------------------------
# Helpers
# -----------------------------
def _to_date_str(x: str) -> str:
    x = x.strip()
    if len(x) == 4:
        return f"{x}-01-01"
    if len(x) == 7:
        return f"{x}-01"
    return x


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class Inputs:
    start: str
    end: str | None
    freq: str  # 'D' or 'M'
    window: int
    neg_threshold: float
    outdir: Path
    save: bool
    # peak/trough detection
    peak_window: int
    match_window: int
    top_n: int


# -----------------------------
# Core logic
# -----------------------------
def load_series(start: str, end: str | None) -> pd.DataFrame:
    """
    Columns:
      - USDKRW: USD/KRW close
      - KOSPI:  KOSPI index close
    """
    fx = fdr.DataReader("USD/KRW", start, end)[["Close"]].rename(columns={"Close": "USDKRW"})
    kospi = fdr.DataReader("KS11", start, end)[["Close"]].rename(columns={"Close": "KOSPI"})
    df = fx.join(kospi, how="inner").dropna()
    if df.empty or df.shape[0] < 50:
        raise ValueError("데이터가 충분하지 않습니다. start/end를 확인하세요.")
    return df


def to_returns(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    x = df.copy()
    if freq.upper() == "M":
        x = x.resample("M").last()
    rets = np.log(x).diff().dropna()
    rets.columns = [c + "_RET" for c in rets.columns]
    return rets


def rolling_corr(rets: pd.DataFrame, window: int) -> pd.Series:
    s = rets["USDKRW_RET"].rolling(window).corr(rets["KOSPI_RET"])
    s.name = "ROLL_CORR"
    return s.dropna()


def summarize(rets: pd.DataFrame, roll: pd.Series) -> dict:
    out = {}
    out["corr_full"] = float(rets["USDKRW_RET"].corr(rets["KOSPI_RET"]))
    n = 252 if len(rets) > 260 else max(60, int(len(rets) * 0.25))
    recent = rets.iloc[-n:]
    out["corr_recent"] = float(recent["USDKRW_RET"].corr(recent["KOSPI_RET"]))
    out["pct_roll_negative"] = float((roll < 0).mean())
    out["pct_roll_below_-0.2"] = float((roll < -0.2).mean())
    out["latest_roll_corr"] = float(roll.iloc[-1])
    out["latest_date"] = str(roll.index[-1].date())
    return out


def find_peak_trough_matches(
    df: pd.DataFrame,
    peak_window: int = 20,
    match_window: int = 5,
    top_n: int = 10
) -> pd.DataFrame:
    """
    '코스피 국지적 고점'과 '달러/원 국지적 저점'의 동시/근접 발생을 찾는다.

    - 코스피 고점: t일의 KOSPI가 [t-peak_window, t+peak_window]에서 최대면 peak
    - 달러원 저점: 같은 방식으로 최소면 trough
    - match: 고점일 t 주변 ±match_window 안에 달러원 trough가 존재하면 매칭

    반환: columns = [date_peak, date_trough, days_diff, kospi, usdk]
    """
    x = df.copy()

    # Centered rolling extremum conditions (requires enough data on both sides)
    kos = x["KOSPI"]
    fx = x["USDKRW"]

    # Rolling max/min with center=True
    roll_max = kos.rolling(window=2 * peak_window + 1, center=True).max()
    roll_min = fx.rolling(window=2 * peak_window + 1, center=True).min()

    is_peak = (kos == roll_max)
    is_trough = (fx == roll_min)

    peak_dates = kos[is_peak].dropna().sort_values(ascending=False).head(top_n).index
    trough_dates_set = set(fx[is_trough].dropna().index)

    rows = []
    for d_peak in peak_dates:
        # search nearest trough within ±match_window
        best = None
        for k in range(-match_window, match_window + 1):
            d2 = d_peak + pd.Timedelta(days=k)
            if d2 in trough_dates_set:
                best = d2
                break
        if best is not None:
            rows.append({
                "date_peak": d_peak,
                "date_trough": best,
                "days_diff": (best - d_peak).days,
                "kospi": float(kos.loc[d_peak]),
                "usdk": float(fx.loc[best]),
            })

    return pd.DataFrame(rows).sort_values(by=["date_peak"])


# -----------------------------
# Plots (more intuitive)
# -----------------------------
def plot_dashboard(
    df: pd.DataFrame,
    rets: pd.DataFrame,
    roll: pd.Series,
    matches: pd.DataFrame,
    window: int,
    neg_threshold: float,
    title_suffix: str = ""
) -> plt.Figure:
    """
    3-panel dashboard:
    (1) Levels (KOSPI + USDKRW) with highlighted peak-trough matches
    (2) Rolling correlation with decoupling zones highlighted
    (3) Return scatter with quadrant annotations
    """
    # Normalize for a cleaner comparison
    base = df.iloc[0]
    norm = df / base * 100.0

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.9, 1.0])

    # (1) Levels
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(norm.index, norm["KOSPI"], label="코스피 (정규화, 시작=100)")
    ax1.set_ylabel("코스피 (정규화)")
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(norm.index, norm["USDKRW"], label="달러/원 (정규화, 시작=100)")
    ax1b.set_ylabel("달러/원 (정규화)")

    ax1.set_title(f"레벨 비교 (정규화) {title_suffix}".strip())

    # Mark peak/trough matches
    if not matches.empty:
        # peak on KOSPI axis
        for _, r in matches.iterrows():
            d_peak = pd.to_datetime(r["date_peak"])
            d_trough = pd.to_datetime(r["date_trough"])
            # peak marker
            ax1.scatter(d_peak, norm.loc[d_peak, "KOSPI"], s=45, marker="^")
            # trough marker
            if d_trough in norm.index:
                ax1b.scatter(d_trough, norm.loc[d_trough, "USDKRW"], s=45, marker="v")
            # connect line (visual cue)
            ax1.plot([d_peak, d_trough],
                     [norm.loc[d_peak, "KOSPI"], norm.loc[d_peak, "KOSPI"]],
                     linewidth=0.8, alpha=0.6)

        ax1.text(0.01, 0.02,
                 "표시: ▲ 코스피 국지적 고점 / ▼ 달러/원 국지적 저점(근접)",
                 transform=ax1.transAxes, fontsize=10, alpha=0.9)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    # (2) Rolling correlation
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(roll.index, roll.values, label=f"롤링 상관 ({window})")
    ax2.axhline(0.0, linewidth=1.0)
    ax2.axhline(neg_threshold, linewidth=1.0)
    below = roll < neg_threshold
    ax2.fill_between(roll.index, roll.values, neg_threshold, where=below, alpha=0.25)
    ax2.set_ylabel("상관계수")
    ax2.set_title(f"달러/원-코스피 롤링 상관 (디커플링 영역 음영) {title_suffix}".strip())
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")

    # (3) Scatter
    ax3 = fig.add_subplot(gs[2, 0])
    x = rets["USDKRW_RET"].values
    y = rets["KOSPI_RET"].values
    ax3.scatter(x, y, alpha=0.35)
    # regression line (intuition)
    if len(x) > 2:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        ax3.plot(xs, b1 * xs + b0, linewidth=1.5)
    ax3.axhline(0.0, linewidth=1.0, alpha=0.7)
    ax3.axvline(0.0, linewidth=1.0, alpha=0.7)

    corr = float(np.corrcoef(x, y)[0, 1])
    ax3.set_title(f"수익률 산점도 (상관={corr:.3f}) {title_suffix}".strip())
    ax3.set_xlabel("달러/원 로그수익률")
    ax3.set_ylabel("코스피 로그수익률")
    ax3.grid(True, alpha=0.3)

    # Quadrant hints
    ax3.text(0.98, 0.98, "원화 약세 & 주식↑\n(예외/레짐 전환)",
             transform=ax3.transAxes, ha="right", va="top", fontsize=9, alpha=0.8)
    ax3.text(0.02, 0.98, "원화 강세 & 주식↑\n(Risk-On 전형)",
             transform=ax3.transAxes, ha="left", va="top", fontsize=9, alpha=0.8)
    ax3.text(0.02, 0.02, "원화 강세 & 주식↓\n(성장둔화/이익 쇼크)",
             transform=ax3.transAxes, ha="left", va="bottom", fontsize=9, alpha=0.8)
    ax3.text(0.98, 0.02, "원화 약세 & 주식↓\n(Risk-Off 전형)",
             transform=ax3.transAxes, ha="right", va="bottom", fontsize=9, alpha=0.8)

    fig.tight_layout()
    return fig


# -----------------------------
# CLI / Main
# -----------------------------
def parse_args() -> Inputs:
    p = argparse.ArgumentParser(
        description="USD/KRW vs KOSPI: 음(-)의 상관(디커플링) 분석 (FinanceDataReader) - 한글/직관형 대시보드"
    )
    p.add_argument("--start", type=str, default="2010-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), default: today")
    p.add_argument("--freq", type=str, default="D", choices=["D", "M"], help="Return frequency: D(daily) or M(monthly)")
    p.add_argument("--window", type=int, default=60, help="Rolling window (days if D, months if M)")
    p.add_argument("--neg_threshold", type=float, default=-0.2, help="Decoupling threshold for rolling corr")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory for figures")
    p.add_argument("--save", action="store_true", help="Save figures to outdir")

    # peak/trough detection knobs
    p.add_argument("--peak_window", type=int, default=20, help="Local-extrema window (days) for peaks/troughs")
    p.add_argument("--match_window", type=int, default=5, help="Allow ±match_window days to match peak vs trough")
    p.add_argument("--top_n", type=int, default=10, help="Mark top N KOSPI peaks to search matches")

    a = p.parse_args()
    return Inputs(
        start=_to_date_str(a.start),
        end=_to_date_str(a.end) if a.end else None,
        freq=a.freq.upper(),
        window=a.window,
        neg_threshold=float(a.neg_threshold),
        outdir=Path(a.outdir),
        save=bool(a.save),
        peak_window=int(a.peak_window),
        match_window=int(a.match_window),
        top_n=int(a.top_n),
    )


def main() -> None:
    setup_korean_font()
    args = parse_args()

    df = load_series(args.start, args.end)
    rets = to_returns(df, freq=args.freq)
    roll = rolling_corr(rets, window=args.window)
    summary = summarize(rets, roll)

    # peak/trough matches are computed on daily levels (original df)
    matches = find_peak_trough_matches(
        df=df,
        peak_window=args.peak_window,
        match_window=args.match_window,
        top_n=args.top_n,
    )

    # --- Print PM-style summary ---
    print("\n[PM-style summary]")
    print(f"- Sample: {rets.index.min().date()} ~ {rets.index.max().date()}  (n={len(rets)})  freq={args.freq}")
    print(f"- Corr (full):    {summary['corr_full']:.3f}")
    print(f"- Corr (recent):  {summary['corr_recent']:.3f}")
    print(f"- Rolling corr < 0:     {summary['pct_roll_negative']*100:.1f}%")
    print(f"- Rolling corr < -0.2:  {summary['pct_roll_below_-0.2']*100:.1f}%")
    print(f"- Latest rolling corr:  {summary['latest_roll_corr']:.3f}  @ {summary['latest_date']}")

    if matches.empty:
        print("\n[피크-트로프 매칭] 조건을 만족하는 '코스피 고점 & 달러/원 저점' 근접 사례를 찾지 못했습니다.")
        print("  - peak_window / match_window를 조정해 보세요. 예: --peak_window 30 --match_window 10")
    else:
        print("\n[피크-트로프 매칭: 코스피 고점(▲) & 달러/원 저점(▼) 근접]")
        print(matches[["date_peak", "date_trough", "days_diff", "kospi", "usdk"]].to_string(index=False))

    title_suffix = f"(freq={args.freq}, start={args.start})"
    fig = plot_dashboard(
        df=df,
        rets=rets,
        roll=roll,
        matches=matches,
        window=args.window,
        neg_threshold=args.neg_threshold,
        title_suffix=title_suffix,
    )

    if args.save:
        _ensure_dir(args.outdir)
        outpath = args.outdir / f"dashboard_{args.freq}_w{args.window}.png"
        fig.savefig(outpath, dpi=180)
        print(f"\n[saved] {outpath}")

    plt.show()


if __name__ == "__main__":
    main()
