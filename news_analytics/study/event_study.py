from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from news_analytics.analytics.calendar import trading_days_from_prices
from news_analytics.analytics.events import chronological_split
from news_analytics.analytics.returns import (
    build_return_panels,
    execution_abnormal_return,
    market_model_abnormal,
)
from news_analytics.db import get_connection

WINDOWS = (("next_day", 1), ("days_1_3", 3), ("days_1_5", 5))


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    mean_diff = float(a.mean() - b.mean())
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df_num = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2
    df_den = (a.var(ddof=1) ** 2) / ((len(a) ** 2) * (len(a) - 1)) + (b.var(ddof=1) ** 2) / (
        (len(b) ** 2) * (len(b) - 1)
    )
    df = df_num / df_den if df_den else np.nan
    tcrit = stats.t.ppf(0.975, df) if np.isfinite(df) else np.nan
    ci_low = mean_diff - tcrit * se if np.isfinite(tcrit) else np.nan
    ci_high = mean_diff + tcrit * se if np.isfinite(tcrit) else np.nan
    return float(t), float(p), float(ci_low), float(ci_high)


def _clustered_se_pvalue(event_ar: pd.DataFrame) -> tuple[float | None, str]:
    """
    OLS of AR on event dummy with ticker-clustered variance (Liang-Zeger style).
    Remaining limitation: overlapping windows still induce date clustering.
    """
    if event_ar.empty or event_ar["ticker"].nunique() < 3:
        return None, "insufficient_clusters"
    y = event_ar["abnormal_return"].to_numpy()
    x = np.column_stack([np.ones(len(event_ar)), event_ar["is_event"].to_numpy()])
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, "lstsq_failed"
    resid = y - x @ beta
    clusters = event_ar["ticker"].to_numpy()
    meat = np.zeros((2, 2))
    for ticker in np.unique(clusters):
        idx = clusters == ticker
        xe = x[idx]
        ue = resid[idx]
        score = xe.T @ ue
        meat += np.outer(score, score)
    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return None, "singular"
    vcov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(max(vcov[1, 1], 0.0))
    if se == 0:
        return None, "zero_se"
    z = beta[1] / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(p), "ticker_clustered"


def run_event_study(db_path: Path, benchmark: str, train_fraction: float = 0.7) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        events = pd.read_sql_query(
            "SELECT * FROM market_events WHERE event_type = 'HIGH_IMPACT'", conn
        )
        features = pd.read_sql_query("SELECT ticker, date, event_score, article_count FROM daily_features", conn)
        prices = pd.read_sql_query("SELECT ticker, trade_date, adj_close FROM stock_prices", conn)
        sectors = pd.read_sql_query("SELECT ticker, sector FROM tickers", conn)

    notes = []
    if events.empty or prices.empty:
        notes.append("Insufficient events or prices; no event study computed.")
        _store_results(db_path, [], pd.DataFrame(), "; ".join(notes))
        return pd.DataFrame()

    events["date"] = pd.to_datetime(events["date"])
    features["date"] = pd.to_datetime(features["date"])
    panels, bench_rets = build_return_panels(prices, benchmark)
    trading_days = trading_days_from_prices(prices, benchmark)
    train_end = chronological_split(features["date"], train_fraction)

    eligible = features[features["article_count"] > 0].copy()
    event_keys = set(zip(events["company"], events["date"]))
    eligible["is_event"] = [
        1 if (t, d) in event_keys else 0 for t, d in zip(eligible["ticker"], eligible["date"])
    ]
    eligible["split"] = np.where(eligible["date"] <= train_end, "train", "test")

    obs_rows = []
    for r in eligible.itertuples(index=False):
        stock = panels.get(r.ticker)
        if stock is None or bench_rets.empty:
            continue
        for window_name, horizon in WINDOWS:
            ar = execution_abnormal_return(stock, bench_rets, r.date, trading_days, horizon=horizon)
            obs_rows.append(
                {
                    "ticker": r.ticker,
                    "event_date": pd.Timestamp(r.date).date().isoformat(),
                    "execution_date": None,
                    "window_name": window_name,
                    "is_event": int(r.is_event),
                    "abnormal_return": ar,
                    "split": r.split,
                }
            )
            if window_name == "next_day" and r.is_event:
                mm = market_model_abnormal(stock, bench_rets, next_exec(trading_days, r.date))
                if mm is not None:
                    obs_rows.append(
                        {
                            "ticker": r.ticker,
                            "event_date": pd.Timestamp(r.date).date().isoformat(),
                            "execution_date": None,
                            "window_name": "next_day_market_model",
                            "is_event": 1,
                            "abnormal_return": mm,
                            "split": r.split,
                        }
                    )

    obs = pd.DataFrame(obs_rows)
    if obs.empty:
        _store_results(
            db_path,
            [],
            obs,
            "No event-study observations (missing prices or returns).",
        )
        return pd.DataFrame()
    summaries = []
    sector_map = dict(zip(sectors["ticker"], sectors["sector"])) if not sectors.empty else {}

    for split in ("train", "test"):
        for window_name, _ in WINDOWS + (("next_day_market_model", 1),):
            sub = obs[(obs["split"] == split) & (obs["window_name"] == window_name)].dropna(
                subset=["abnormal_return"]
            )
            if sub.empty:
                continue
            ev = sub[sub["is_event"] == 1]["abnormal_return"].to_numpy()
            ctrl = sub[sub["is_event"] == 0]["abnormal_return"].to_numpy()
            if window_name == "next_day_market_model":
                ctrl = np.array([])
            t, p, lo, hi = _welch(ev, ctrl) if len(ctrl) else (np.nan, np.nan, np.nan, np.nan)
            mwu_u = mwu_p = np.nan
            if len(ev) >= 3 and len(ctrl) >= 3:
                try:
                    mwu_u, mwu_p = stats.mannwhitneyu(ev, ctrl, alternative="two-sided")
                except ValueError:
                    pass
            n_tickers = int(sub.loc[sub["is_event"] == 1, "ticker"].nunique()) if len(ev) else 0
            clustered_p, clustered_note = _clustered_se_pvalue(sub) if window_name == "next_day" else (None, "")
            note = (
                "Welch t-test assumes independent samples with possibly unequal variance; "
                "news days are clustered by ticker and calendar date, so p-values are anti-conservative. "
                f"Ticker-clustered dummy-regression p={clustered_p}. {clustered_note}. "
                "Multiple windows/splits inflate false-positive risk; do not treat p<0.05 as a guarantee."
            )
            summaries.append(
                {
                    "split": split,
                    "window_name": window_name,
                    "n_events": int(len(ev)),
                    "n_tickers": n_tickers,
                    "mean_ar": float(np.mean(ev)) if len(ev) else None,
                    "median_ar": float(np.median(ev)) if len(ev) else None,
                    "mean_ar_control": float(np.mean(ctrl)) if len(ctrl) else None,
                    "median_ar_control": float(np.median(ctrl)) if len(ctrl) else None,
                    "diff_mean": float(np.mean(ev) - np.mean(ctrl)) if len(ev) and len(ctrl) else None,
                    "ci_low": None if np.isnan(lo) else float(lo),
                    "ci_high": None if np.isnan(hi) else float(hi),
                    "welch_t": None if np.isnan(t) else float(t),
                    "welch_p": None if np.isnan(p) else float(p),
                    "mwu_u": None if np.isnan(mwu_u) else float(mwu_u),
                    "mwu_p": None if np.isnan(mwu_p) else float(mwu_p),
                    "notes": note,
                }
            )
            if window_name == "next_day" and split == "test" and len(ev):
                tmp = sub[sub["is_event"] == 1].copy()
                tmp["sector"] = tmp["ticker"].map(sector_map)
                for sector, g in tmp.groupby("sector"):
                    if not sector or g.empty:
                        continue
                    summaries.append(
                        {
                            "split": f"test_sector:{sector}",
                            "window_name": window_name,
                            "n_events": int(len(g)),
                            "n_tickers": int(g["ticker"].nunique()),
                            "mean_ar": float(g["abnormal_return"].mean()),
                            "median_ar": float(g["abnormal_return"].median()),
                            "mean_ar_control": None,
                            "median_ar_control": None,
                            "diff_mean": None,
                            "ci_low": None,
                            "ci_high": None,
                            "welch_t": None,
                            "welch_p": None,
                            "mwu_u": None,
                            "mwu_p": None,
                            "notes": "Descriptive sector slice only; not a pre-registered hypothesis test.",
                        }
                    )

    _store_results(db_path, summaries, obs, "")
    return pd.DataFrame(summaries)


def next_exec(trading_days: pd.DatetimeIndex, publish_date):
    from news_analytics.analytics.calendar import next_trading_day

    return next_trading_day(trading_days, publish_date)


def _store_results(db_path: Path, summaries: list[dict], obs: pd.DataFrame, fallback_note: str) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM event_study_results;")
        conn.execute("DELETE FROM event_study_observations;")
        if summaries:
            conn.executemany(
                """
                INSERT INTO event_study_results (
                    split, window_name, n_events, n_tickers, mean_ar, median_ar,
                    mean_ar_control, median_ar_control, diff_mean, ci_low, ci_high,
                    welch_t, welch_p, mwu_u, mwu_p, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s["split"],
                        s["window_name"],
                        s["n_events"],
                        s["n_tickers"],
                        s["mean_ar"],
                        s["median_ar"],
                        s["mean_ar_control"],
                        s["median_ar_control"],
                        s["diff_mean"],
                        s["ci_low"],
                        s["ci_high"],
                        s["welch_t"],
                        s["welch_p"],
                        s["mwu_u"],
                        s["mwu_p"],
                        s["notes"],
                    )
                    for s in summaries
                ],
            )
        elif fallback_note:
            conn.execute(
                """
                INSERT INTO event_study_results (split, window_name, n_events, notes)
                VALUES ('none', 'none', 0, ?)
                """,
                (fallback_note,),
            )
        if not obs.empty:
            conn.executemany(
                """
                INSERT OR REPLACE INTO event_study_observations
                    (ticker, event_date, execution_date, window_name, is_event, abnormal_return, split)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.ticker,
                        r.event_date,
                        r.execution_date,
                        r.window_name,
                        int(r.is_event),
                        None if pd.isna(r.abnormal_return) else float(r.abnormal_return),
                        r.split,
                    )
                    for r in obs.itertuples(index=False)
                ],
            )
