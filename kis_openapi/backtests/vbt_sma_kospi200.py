from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd
import vectorbt as vbt

PKG_DIR = Path(__file__).resolve().parents[1]  # .../kis_openapi
WS_DIR = Path(__file__).resolve().parents[2]  # workspace root
if str(WS_DIR) not in sys.path:
    sys.path.insert(0, str(WS_DIR))

from kis_openapi.kis import KISClient, KISConfig
from kis_openapi.kis.marketdata import fetch_kospi200_daily
from kis_openapi.kis.utils import load_dotenv_if_present


def main() -> None:
    load_dotenv_if_present(PKG_DIR / ".env")
    cfg = KISConfig.from_env()
    client = KISClient(cfg)

    df = fetch_kospi200_daily(client, cfg, days=365)

    close = pd.Series(
        df["close"].to_numpy(),
        index=pd.to_datetime(df["date"]),
        name="KOSPI200",
    ).sort_index()

    fast_window = int(os.getenv("VBT_FAST", "20"))
    slow_window = int(os.getenv("VBT_SLOW", "60"))

    fast = vbt.MA.run(close, window=fast_window)
    slow = vbt.MA.run(close, window=slow_window)

    entries = fast.ma_crossed_above(slow)
    exits = fast.ma_crossed_below(slow)

    fees = float(os.getenv("VBT_FEES", "0.00015"))
    slippage = float(os.getenv("VBT_SLIPPAGE", "0.0005"))

    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        fees=fees,
        slippage=slippage,
        init_cash=float(os.getenv("VBT_INIT_CASH", "10000000")),
        freq="1D",
    )

    print(pf.stats())

    # Plotly 6 defaults to FigureWidget in some contexts; avoid anywidget dependency.
    vbt.settings["plotting"]["use_widgets"] = False

    out_dir = PKG_DIR / "data" / "reports" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = (out_dir / "output_vbt_kospi200_sma.html").as_posix()
    fig = pf.plot()
    # fig can be plotly.graph_objects.Figure
    fig.write_html(out_html)
    print(f"saved: {out_html}")


if __name__ == "__main__":
    main()
