"""21 trading-day simple return and SPY excess — aligned with notebook 03 definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_TRADING_DAYS_1M = 21


def mean_ret_excess_by_sentiment(
    df: pd.DataFrame,
    group_col: str,
    *,
    label_order: List[str],
) -> pd.DataFrame:
    """Rows indexed by ``label_order`` with n, mean_ret_1m, mean_excess_1m."""
    sub = df[df["ret_1m"].notna() & df["excess_1m"].notna()].copy()
    g = (
        sub.groupby(group_col, dropna=False)
        .agg(
            n=("row_id", "count"),
            mean_ret_1m=("ret_1m", "mean"),
            mean_excess_1m=("excess_1m", "mean"),
        )
        .reindex(label_order)
    )
    return g


def load_adj_close_panel(raw_dir: Path) -> pd.DataFrame:
    """Concat daily equity + SPY CSVs under data/raw/."""
    parts = [
        pd.read_csv(raw_dir / "daily_prices_2010_2014.csv", parse_dates=["Date"]),
        pd.read_csv(raw_dir / "daily_prices_2015_2026.csv", parse_dates=["Date"]),
        pd.read_csv(raw_dir / "spy_daily_2010_2026.csv", parse_dates=["Date"]),
    ]
    df = pd.concat(parts, ignore_index=True)
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def forward_simple_return(
    ticker_rows: pd.DataFrame,
    event_date: pd.Timestamp,
    horizon: int,
) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    evt = pd.Timestamp(event_date).normalize()
    sub = ticker_rows[ticker_rows["Date"] >= evt].sort_values("Date")
    if len(sub) < horizon + 1:
        return np.nan, pd.NaT, pd.NaT
    t0 = sub.iloc[0]["Date"]
    t1 = sub.iloc[horizon]["Date"]
    p0 = float(sub.iloc[0]["Adj Close"])
    p1 = float(sub.iloc[horizon]["Adj Close"])
    if p0 <= 0 or np.isnan(p0) or np.isnan(p1):
        return np.nan, t0, t1
    return p1 / p0 - 1.0, t0, t1


def spy_return_on_dates(spy_rows: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    s = spy_rows.set_index("Date")["Adj Close"].sort_index()
    idx = s.index

    def price_on(d: pd.Timestamp) -> float:
        if d in s.index:
            return float(s.loc[d])
        pos = idx.searchsorted(d)
        if pos >= len(idx):
            return np.nan
        return float(s.iloc[pos])

    p0, p1 = price_on(t0), price_on(t1)
    if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
        return np.nan
    return p1 / p0 - 1.0


def attach_21d_return_and_excess(
    df: pd.DataFrame,
    *,
    raw_dir: Path,
    ticker_col: str = "ticker",
    date_col: str = "parsed_report_date",
    horizon: int = DEFAULT_TRADING_DAYS_1M,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``ret_1m`` and ``excess_1m`` columns."""
    prices = load_adj_close_panel(raw_dir)
    spy_rows = prices[prices["Ticker"] == "SPY"][["Date", "Adj Close"]].copy()
    by_ticker: Dict[str, pd.DataFrame] = {t: g for t, g in prices.groupby("Ticker")}

    rets: list[float] = []
    excess: list[float] = []
    for _, row in df.iterrows():
        tk = str(row[ticker_col]).upper()
        if tk not in by_ticker:
            rets.append(np.nan)
            excess.append(np.nan)
            continue
        r_stock, t0, t1 = forward_simple_return(by_ticker[tk], row[date_col], horizon)
        if np.isnan(r_stock) or t0 is pd.NaT:
            rets.append(np.nan)
            excess.append(np.nan)
            continue
        r_spy = spy_return_on_dates(spy_rows, t0, t1)
        if np.isnan(r_spy):
            rets.append(r_stock)
            excess.append(np.nan)
            continue
        rets.append(r_stock)
        excess.append(r_stock - r_spy)

    out = df.copy()
    out["ret_1m"] = rets
    out["excess_1m"] = excess
    return out
