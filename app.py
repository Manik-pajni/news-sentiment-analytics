"""Streamlit dashboard. Launch: streamlit run app.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from news_analytics.config import load_settings
from news_analytics.reporting import mapping_stats, overview, q

st.set_page_config(page_title="News Sentiment Analytics", layout="wide")

FORMULAS = {
    "sentiment": "Primary sentiment is GDELT AvgTone / 10, clipped to [-1, 1]. It is not VADER.",
    "volume_z": "Causal 20-session z-score of article_count vs the same ticker's history. Null if std≈0 or short history.",
    "event_score": "mean_sentiment × max(volume_z_20, 0). Threshold = 90th percentile of train-period scores.",
    "abnormal_return": "Next-trading-day stock return minus benchmark return. Not a raw return.",
    "sharpe": "sqrt(252) × mean(daily portfolio return) / std(daily portfolio return), rf = 0.",
}


def _db() -> Path:
    return load_settings().database_path


def _empty(msg: str) -> None:
    st.info(msg)


def page_overview():
    st.header("Overview")
    if not _db().exists():
        _empty("Database not found. Run `python -m news_analytics.pipeline --fetch` first.")
        return
    stats = overview(_db())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles", stats["n_articles"])
    c2.metric("Mapped mentions", stats["n_mapped_mentions"])
    c3.metric("Equities mapped", stats["n_tickers_mapped"])
    c4.metric("High-impact events", stats["n_high_impact"])
    st.write(
        f"News date range: **{stats['date_min'] or 'n/a'}** → **{stats['date_max'] or 'n/a'}**. "
        f"Price rows: {stats['n_price_rows']}. Last pipeline log: {stats['last_run'] or 'n/a'}."
    )
    st.caption(FORMULAS["event_score"])
    runs = q(_db(), "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 20")
    if runs.empty:
        _empty("No pipeline runs recorded yet.")
    else:
        st.subheader("Recent pipeline steps")
        st.dataframe(runs, use_container_width=True)


def page_news():
    st.header("News Explorer")
    st.caption(FORMULAS["sentiment"])
    if not _db().exists():
        _empty("No database.")
        return
    df = q(
        _db(),
        """
        SELECT na.publish_date, cm.company AS ticker, t.sector, na.sentiment, na.tone,
               cm.match_method, cm.match_confidence, na.source, na.actor1
        FROM news_articles na
        JOIN company_mentions cm ON na.article_id = cm.article_id
        LEFT JOIN tickers t ON t.ticker = cm.company
        """,
    )
    if df.empty:
        _empty("No mapped news yet. Run `python -m news_analytics.pipeline --fetch` to download recent GDELT files.")
        return
    tickers = ["(all)"] + sorted(df["ticker"].dropna().unique().tolist())
    sectors = ["(all)"] + sorted([s for s in df["sector"].dropna().unique().tolist()])
    c1, c2, c3 = st.columns(3)
    ticker = c1.selectbox("Ticker", tickers)
    sector = c2.selectbox("Sector", sectors)
    sent = c3.slider("Sentiment range", -1.0, 1.0, (-1.0, 1.0))
    view = df.copy()
    if ticker != "(all)":
        view = view[view["ticker"] == ticker]
    if sector != "(all)":
        view = view[view["sector"] == sector]
    view = view[(view["sentiment"] >= sent[0]) & (view["sentiment"] <= sent[1])]
    st.dataframe(view.sort_values("publish_date", ascending=False).head(500), use_container_width=True)


def page_market():
    st.header("Market Impact")
    st.caption(FORMULAS["abnormal_return"])
    if not _db().exists():
        _empty("No database.")
        return
    res = q(_db(), "SELECT * FROM event_study_results")
    if res.empty:
        _empty("Event-study results are empty. The pipeline may lack news, prices, or events.")
        return
    st.dataframe(res, use_container_width=True)
    test = res[(res["split"] == "test") & (res["window_name"] == "next_day")]
    if not test.empty:
        row = test.iloc[0]
        st.subheader("Test window: next trading day")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("N events", int(row["n_events"] or 0))
        k2.metric("Mean AR", None if pd.isna(row["mean_ar"]) else f"{row['mean_ar']:.4%}")
        k3.metric("Diff vs control", None if pd.isna(row["diff_mean"]) else f"{row['diff_mean']:.4%}")
        k4.metric("Welch p", None if pd.isna(row["welch_p"]) else f"{row['welch_p']:.3f}")
        st.write(row.get("notes") or "")
    obs = q(
        _db(),
        """
        SELECT * FROM event_study_observations
        WHERE window_name='next_day' AND is_event=1 AND split='test'
        """,
    )
    if not obs.empty:
        st.bar_chart(obs.set_index("event_date")["abnormal_return"])


def page_ticker():
    st.header("Ticker Analysis")
    st.caption(FORMULAS["volume_z"])
    if not _db().exists():
        _empty("No database.")
        return
    feat = q(_db(), "SELECT * FROM daily_features")
    if feat.empty:
        _empty("No daily features. Run aggregation after ingesting news and prices.")
        return
    ticker = st.selectbox("Ticker", sorted(feat["ticker"].unique().tolist()))
    sub = feat[feat["ticker"] == ticker].sort_values("date")
    st.line_chart(sub.set_index("date")[["mean_sentiment", "rolling_sent_5", "rolling_sent_20"]])
    st.line_chart(sub.set_index("date")[["article_count", "volume_z_20"]])
    ev = q(_db(), "SELECT * FROM market_events WHERE company = ?", (ticker,))
    st.subheader("Events")
    if ev.empty:
        _empty("No events for this ticker.")
    else:
        st.dataframe(ev, use_container_width=True)


def page_backtest():
    st.header("Backtesting")
    st.caption(FORMULAS["sharpe"])
    if not _db().exists():
        _empty("No database.")
        return
    summary = q(_db(), "SELECT * FROM backtest_summary ORDER BY id DESC LIMIT 5")
    daily = q(_db(), "SELECT * FROM backtest_daily ORDER BY date")
    trades = q(_db(), "SELECT * FROM backtest_trades ORDER BY trade_date")
    if summary.empty:
        _empty("No backtest results. This is expected when there are no test-period signals or prices.")
        return
    row = summary.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", None if pd.isna(row["total_return"]) else f"{row['total_return']:.2%}")
    c2.metric("Benchmark", None if pd.isna(row["benchmark_total_return"]) else f"{row['benchmark_total_return']:.2%}")
    c3.metric("Sharpe", None if pd.isna(row["sharpe"]) else f"{row['sharpe']:.2f}")
    c4.metric("Max drawdown", None if pd.isna(row["max_drawdown"]) else f"{row['max_drawdown']:.2%}")
    st.write(
        f"Trades: {row['n_trades']}. Costs: {row['transaction_cost_bps']} bps + "
        f"{row['slippage_bps']} bps slippage. {row['notes']}"
    )
    if not daily.empty:
        st.line_chart(daily.set_index("date")[["cumulative_return", "benchmark_cumulative"]])
        st.line_chart(daily.set_index("date")[["drawdown"]])
    if trades.empty:
        _empty("No trades in the test window.")
    else:
        st.dataframe(trades, use_container_width=True)


def page_quality():
    st.header("Data Quality")
    if not _db().exists():
        _empty("No database.")
        return
    stats = mapping_stats(_db())
    st.write(stats)
    unmatched = q(
        _db(),
        "SELECT * FROM unmatched_mentions ORDER BY mention_count DESC LIMIT 200",
    )
    errors = q(_db(), "SELECT * FROM ingestion_errors ORDER BY id DESC LIMIT 100")
    dups = q(
        _db(),
        """
        SELECT article_id, COUNT(*) AS n FROM news_articles
        GROUP BY article_id HAVING n > 1
        """,
    )
    st.subheader("Unmatched mentions")
    if unmatched.empty:
        _empty("No unmatched mentions stored.")
    else:
        st.dataframe(unmatched, use_container_width=True)
    st.subheader("Ingestion errors")
    if errors.empty:
        _empty("No ingestion errors logged.")
    else:
        st.dataframe(errors, use_container_width=True)
    st.subheader("Duplicate article IDs")
    if dups.empty:
        st.success("No duplicate article_id values (primary key enforced).")
    else:
        st.dataframe(dups)


def main():
    st.sidebar.title("News Sentiment Analytics")
    page = st.sidebar.radio(
        "Section",
        [
            "Overview",
            "News Explorer",
            "Market Impact",
            "Ticker Analysis",
            "Backtesting",
            "Data Quality",
        ],
    )
    {
        "Overview": page_overview,
        "News Explorer": page_news,
        "Market Impact": page_market,
        "Ticker Analysis": page_ticker,
        "Backtesting": page_backtest,
        "Data Quality": page_quality,
    }[page]()


if __name__ == "__main__":
    main()
