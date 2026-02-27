"""
ClawBot Dashboard — Streamlit
==============================
Run with:  streamlit run dashboard/app.py

Pages:
  • Overview    — equity curve, P&L summary, circuit breaker status
  • Positions   — live positions table
  • Strategies  — per-strategy performance stats
  • Trades      — full trade history
  • Signals     — live meme + news signals from crawlers
  • Risk        — safety layer status, circuit breakers, audit log
"""
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone

from config.settings import STARTING_CAPITAL
from core.broker import Broker
from core.portfolio import PortfolioManager
from core.safety import SafetyLayer, BreakerLevel
from core.audit import read_audit_log
from core.optimizer import Optimizer, PARAM_BOUNDS
from crawler.signal_aggregator import SignalAggregator

# One portfolio per strategy — must match clawbot_cli.py STRATEGY_PORTFOLIOS
STRATEGY_PORTFOLIOS = [
    "meme_momentum",
    "technical_trend",
    "mean_reversion",
    "options_flow",
    "macro_news",
]
COMBINED_START = STARTING_CAPITAL * len(STRATEGY_PORTFOLIOS)   # 5 × $100 k = $500 k

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ClawBot Dashboard",
    page_icon="🦞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-level singletons (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_components():
    from strategies.meme_momentum import MemeMomentumStrategy
    from strategies.technical_trend import TechnicalTrendStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.options_flow import OptionsFlowStrategy
    from strategies.macro_news import MacroNewsStrategy

    # One isolated broker per strategy portfolio
    strategy_brokers = {pid: Broker(portfolio_id=pid) for pid in STRATEGY_PORTFOLIOS}

    # PortfolioManager used read-only (get_equity_curve, get_trade_history,
    # get_strategy_stats).  We pass any broker — these methods query the shared DB
    # regardless of portfolio_id.  We NEVER call .sync() from the dashboard.
    portfolio = PortfolioManager(strategy_brokers["meme_momentum"])

    safety    = SafetyLayer()
    aggregator = SignalAggregator()
    strategies = [
        MemeMomentumStrategy(),
        TechnicalTrendStrategy(),
        MeanReversionStrategy(),
        OptionsFlowStrategy(),
        MacroNewsStrategy(),
    ]
    optimizer = Optimizer(strategies, aggregator)
    return strategy_brokers, portfolio, safety, aggregator, optimizer


strategy_brokers, portfolio, safety, aggregator, optimizer = get_components()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.image("https://em-content.zobj.net/source/apple/391/crab_1f980.png", width=80)
st.sidebar.title("ClawBot")
st.sidebar.caption("Paper Trading Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Positions", "Strategies", "Trade History", "Live Signals",
     "Market Data", "Self-Improvement", "Risk & Safety"],
    index=0,
)

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
if auto_refresh:
    import time
    st.sidebar.caption(f"Next refresh in ~30s")

# Manual refresh button
if st.sidebar.button("Refresh Now"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Helper: sync portfolio
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def get_combined_state() -> dict:
    """
    Aggregate live account + position data across all 5 strategy portfolios.
    Pure reads (get_account / get_positions) — no DB writes.
    Returns a plain dict so @st.cache_data can serialise it.
    """
    total_value = total_cash = total_equity = 0.0
    all_positions: list[dict] = []
    per_strategy: dict[str, dict] = {}

    for pid in STRATEGY_PORTFOLIOS:
        acct      = strategy_brokers[pid].get_account()
        positions = strategy_brokers[pid].get_positions()
        per_strategy[pid] = {
            "portfolio_value": acct.portfolio_value,
            "cash":            acct.cash,
            "equity":          acct.equity,
            "pnl":             acct.portfolio_value - STARTING_CAPITAL,
            "position_count":  len(positions),
        }
        total_value  += acct.portfolio_value
        total_cash   += acct.cash
        total_equity += acct.equity
        for pos in positions:
            all_positions.append({
                "strategy":          pid,
                "symbol":            pos.symbol,
                "qty":               pos.qty,
                "avg_entry_price":   pos.avg_entry_price,
                "current_price":     pos.current_price,
                "market_value":      pos.market_value,
                "unrealized_pnl":    pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
            })

    return {
        "portfolio_value": total_value,
        "cash":            total_cash,
        "equity":          total_equity,
        "total_pnl":       total_value - COMBINED_START,
        "position_count":  len(all_positions),
        "all_positions":   all_positions,
        "per_strategy":    per_strategy,
    }


@st.cache_data(ttl=30)
def get_equity_curve() -> dict:
    """
    Read all portfolio snapshots (written by run-cycle, not the dashboard).
    Uses get_equity_curve() which already returns portfolio_id per record,
    then splits into combined + per-strategy curves in the dashboard.
    Avoids depending on any method added after the cache_resource was created.
    """
    raw = portfolio.get_equity_curve(limit=2000)
    if not raw:
        return {"combined": [], "per_strategy": {}}

    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["bucket"] = df["timestamp"].dt.floor("5min")

    # Per-strategy split (portfolio_id column may be absent on old schema)
    per_strategy: dict[str, list[dict]] = {}
    has_pid = "portfolio_id" in df.columns
    for pid in STRATEGY_PORTFOLIOS:
        df_s = df[df["portfolio_id"] == pid] if has_pid else pd.DataFrame()
        if df_s.empty:
            continue
        grp = (df_s.groupby("bucket")["portfolio_value"]
                   .last().reset_index()
                   .rename(columns={"bucket": "timestamp"})
                   .tail(200))
        per_strategy[pid] = grp.to_dict("records")

    # Combined: sum all strategies per time bucket
    combined = (df.groupby("bucket")["portfolio_value"]
                  .sum().reset_index()
                  .rename(columns={"bucket": "timestamp"})
                  .tail(200))

    return {"combined": combined.to_dict("records"), "per_strategy": per_strategy}


@st.cache_data(ttl=30)
def get_trade_history():
    return portfolio.get_trade_history(limit=200)

@st.cache_data(ttl=30)
def get_strategy_stats():
    return portfolio.get_strategy_stats()

@st.cache_data(ttl=60)
def get_signals():
    return aggregator.refresh()

@st.cache_resource
def get_market_db():
    from core.market_db import MarketDatabase
    return MarketDatabase()

@st.cache_data(ttl=3600)
def get_market_stats():
    return get_market_db().get_stats()

@st.cache_data(ttl=3600)
def get_correlation_matrix_cached(symbols_key: str, period: int):
    symbols = symbols_key.split(",")
    return get_market_db().get_correlation_matrix(symbols, period_days=period)

@st.cache_data(ttl=3600)
def get_top_pairs_cached(period: int, min_r: float):
    return get_market_db().get_all_correlation_pairs(period_days=period, min_abs_r=min_r)

@st.cache_data(ttl=3600)
def get_price_bars_cached(symbol: str, days: int):
    df = get_market_db().get_bars(symbol, days=days)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

# ---------------------------------------------------------------------------
# Circuit Breaker Banner
# ---------------------------------------------------------------------------

cb_state = safety.get_state()
if cb_state.level == BreakerLevel.LEVEL_3:
    st.error(f"🛑 **CIRCUIT BREAKER LEVEL 3 — FULL HALT** | {cb_state.triggered_reason}")
elif cb_state.level == BreakerLevel.LEVEL_2:
    st.warning(f"⚠️ **Circuit Breaker Level 2 — No New Orders** | {cb_state.triggered_reason}")
elif cb_state.level == BreakerLevel.LEVEL_1:
    st.warning(f"🟡 **Circuit Breaker Level 1 — Position Sizes Halved** | {cb_state.triggered_reason}")
else:
    st.success("✅ All systems normal")

# ---------------------------------------------------------------------------
# OVERVIEW PAGE
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("Portfolio Overview")

    state = get_combined_state()

    # --- Combined key metrics ---
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Combined Value",     f"${state['portfolio_value']:,.2f}",
                f"{state['portfolio_value'] - COMBINED_START:+,.2f}")
    col2.metric("Combined Cash",      f"${state['cash']:,.2f}")
    col3.metric("Combined Equity",    f"${state['equity']:,.2f}")
    col4.metric("Total P&L",          f"${state['total_pnl']:+,.2f}",
                f"{state['total_pnl'] / COMBINED_START:+.2%}")
    col5.metric("Open Positions",     state["position_count"])

    # --- Per-strategy portfolio cards ---
    st.divider()
    st.subheader("Per-Strategy Portfolios")
    s_cols = st.columns(len(STRATEGY_PORTFOLIOS))
    for col, pid in zip(s_cols, STRATEGY_PORTFOLIOS):
        s = state["per_strategy"][pid]
        col.metric(
            pid.replace("_", " ").title(),
            f"${s['portfolio_value']:,.0f}",
            f"${s['pnl']:+,.0f}",
        )

    st.divider()

    # --- Equity curves ---
    curve_data = get_equity_curve()
    combined_curve   = curve_data["combined"]
    per_strat_curves = curve_data["per_strategy"]

    _STRATEGY_COLORS = {
        "meme_momentum":   "#ff6b6b",
        "technical_trend": "#4ecdc4",
        "mean_reversion":  "#45b7d1",
        "options_flow":    "#96ceb4",
        "macro_news":      "#feca57",
    }

    tab_combined, tab_per_strat = st.tabs(["Combined", "Per-Strategy"])

    with tab_combined:
        if combined_curve:
            df_curve = pd.DataFrame(combined_curve)
            df_curve["timestamp"] = pd.to_datetime(df_curve["timestamp"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_curve["timestamp"], y=df_curve["portfolio_value"],
                mode="lines", name="Combined",
                line=dict(color="#00b4d8", width=2),
                fill="tozeroy", fillcolor="rgba(0,180,216,0.1)",
            ))
            fig.add_hline(y=COMBINED_START, line_dash="dash", line_color="gray",
                          annotation_text=f"Starting Capital (${COMBINED_START:,.0f})")
            fig.update_layout(
                title="Combined Portfolio Value (all 5 strategies)",
                xaxis_title="Time", yaxis_title="Portfolio Value ($)",
                height=350, margin=dict(l=0, r=0, t=40, b=0), template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No equity history yet — run a cycle to begin recording.")

    with tab_per_strat:
        if per_strat_curves:
            fig2 = go.Figure()
            for pid in STRATEGY_PORTFOLIOS:
                records = per_strat_curves.get(pid)
                if not records:
                    continue
                df_s = pd.DataFrame(records)
                df_s["timestamp"] = pd.to_datetime(df_s["timestamp"])
                fig2.add_trace(go.Scatter(
                    x=df_s["timestamp"], y=df_s["portfolio_value"],
                    mode="lines", name=pid.replace("_", " ").title(),
                    line=dict(color=_STRATEGY_COLORS.get(pid, "#adb5bd"), width=1.5),
                ))
            fig2.add_hline(y=STARTING_CAPITAL, line_dash="dash", line_color="gray",
                           annotation_text=f"${STARTING_CAPITAL:,.0f}/strategy")
            fig2.update_layout(
                title="Per-Strategy Portfolio Value",
                xaxis_title="Time", yaxis_title="Portfolio Value ($)",
                height=350, margin=dict(l=0, r=0, t=40, b=0), template="plotly_dark",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No per-strategy history yet — run a cycle to begin recording.")

    # Strategy P&L pie
    stats = get_strategy_stats()
    if stats:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Strategy P&L")
            df_stats = pd.DataFrame(stats)
            if not df_stats.empty and df_stats["total_pnl"].abs().sum() > 0:
                fig_pie = px.bar(
                    df_stats, x="strategy", y="total_pnl",
                    color="total_pnl",
                    color_continuous_scale=["#e63946", "#adb5bd", "#2a9d8f"],
                    title="P&L by Strategy",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.subheader("Win Rate by Strategy")
            if not df_stats.empty:
                fig_wr = px.bar(
                    df_stats, x="strategy", y="win_rate",
                    color="win_rate",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 100],
                    title="Win Rate (%)",
                    template="plotly_dark",
                )
                fig_wr.add_hline(y=50, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_wr, use_container_width=True)

# ---------------------------------------------------------------------------
# POSITIONS PAGE
# ---------------------------------------------------------------------------

elif page == "Positions":
    st.title("Open Positions")
    state = get_combined_state()

    if not state["all_positions"]:
        st.info("No open positions across any strategy portfolio.")
    else:
        rows = [
            {
                "Strategy":       p["strategy"].replace("_", " ").title(),
                "Symbol":         p["symbol"],
                "Qty":            p["qty"],
                "Entry Price":    f"${p['avg_entry_price']:.2f}",
                "Current Price":  f"${p['current_price']:.2f}",
                "Market Value":   f"${p['market_value']:,.2f}",
                "Unrealized P&L": f"${p['unrealized_pnl']:+,.2f}",
                "P&L %":          f"{p['unrealized_pnl_pct']:+.2%}",
            }
            for p in state["all_positions"]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Position bar chart coloured by strategy
        df_chart = pd.DataFrame([
            {
                "Strategy":     p["strategy"].replace("_", " ").title(),
                "Symbol":       p["symbol"],
                "Market Value": p["market_value"],
            }
            for p in state["all_positions"]
        ])
        fig = px.bar(
            df_chart, x="Symbol", y="Market Value", color="Strategy",
            title="Position Sizes by Strategy", template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# STRATEGIES PAGE
# ---------------------------------------------------------------------------

elif page == "Strategies":
    st.title("Strategy Performance")

    strategy_descriptions = {
        "meme_momentum":    "Social-media momentum (Reddit WSB, StockTwits)",
        "technical_trend":  "EMA trend following + RSI + MACD",
        "mean_reversion":   "Bollinger Band mean reversion + Z-score",
        "options_flow":     "Unusual options activity → equity positioning",
        "macro_news":       "NLP sentiment on earnings, SEC filings, news",
    }

    state = get_combined_state()
    stats = get_strategy_stats()
    # Build a lookup from strategy name → trade stats
    stats_by_name = {s["strategy"]: s for s in stats}

    for pid in STRATEGY_PORTFOLIOS:
        s = state["per_strategy"][pid]
        trade_stat = stats_by_name.get(pid, {})
        label = pid.replace("_", " ").title()
        pnl_delta = f"${s['pnl']:+,.2f}"

        with st.expander(f"**{label}** — ${s['portfolio_value']:,.0f}  ({pnl_delta})", expanded=True):
            # Portfolio financials + trade stats in one row
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Portfolio Value", f"${s['portfolio_value']:,.0f}", pnl_delta)
            c2.metric("Cash",            f"${s['cash']:,.0f}")
            c3.metric("Total Trades",    trade_stat.get("total_trades", 0))
            c4.metric("Win Rate",        f"{trade_stat.get('win_rate', 0):.1f}%")
            c5.metric("Realized P&L",    f"${trade_stat.get('total_pnl', 0):+,.2f}")

            desc = strategy_descriptions.get(pid, "")
            if desc:
                st.caption(desc)

            # Open positions for this strategy
            pos_rows = [p for p in state["all_positions"] if p["strategy"] == pid]
            if pos_rows:
                st.markdown("**Open Positions**")
                pos_df = pd.DataFrame([{
                    "Symbol":         p["symbol"],
                    "Qty":            p["qty"],
                    "Entry":          f"${p['avg_entry_price']:.2f}",
                    "Current":        f"${p['current_price']:.2f}",
                    "Mkt Value":      f"${p['market_value']:,.2f}",
                    "Unrealized P&L": f"${p['unrealized_pnl']:+,.2f}",
                    "P&L %":          f"{p['unrealized_pnl_pct']:+.2%}",
                } for p in pos_rows])
                st.dataframe(pos_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No open positions.")

    if not STRATEGY_PORTFOLIOS:
        st.info("No strategies configured.")

# ---------------------------------------------------------------------------
# TRADE HISTORY PAGE
# ---------------------------------------------------------------------------

elif page == "Trade History":
    st.title("Trade History")

    trades = get_trade_history()
    if not trades:
        st.info("No trades yet.")
    else:
        df_all = pd.DataFrame(trades)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
        df_all["timestamp_fmt"] = df_all["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        df_all["pnl_fmt"] = df_all["pnl"].apply(lambda x: f"${x:+.2f}")
        df_all["meme"]    = df_all["is_meme"].apply(lambda x: "🚀" if x else "")

        display_cols = ["timestamp_fmt", "symbol", "action", "qty", "price",
                        "strategy", "pnl_fmt", "meme"]
        col_rename   = {"timestamp_fmt": "Time", "pnl_fmt": "P&L", "meme": "Meme"}

        # Tabs: All + one per strategy
        tab_labels = ["All"] + [pid.replace("_", " ").title() for pid in STRATEGY_PORTFOLIOS]
        tabs = st.tabs(tab_labels)

        def _render_trades(df: pd.DataFrame, key_suffix: str) -> None:
            if df.empty:
                st.info("No trades for this strategy.")
                return
            st.dataframe(
                df[display_cols].rename(columns=col_rename),
                use_container_width=True, hide_index=True,
            )
            df_pnl = df.copy().sort_values("timestamp")
            df_pnl["cumulative_pnl"] = df_pnl["pnl"].cumsum()
            fig = px.line(df_pnl, x="timestamp_fmt", y="cumulative_pnl",
                          title="Cumulative P&L",
                          template="plotly_dark",
                          color_discrete_sequence=["#2a9d8f"])
            fig.update_xaxes(title="")
            st.plotly_chart(fig, use_container_width=True, key=f"pnl_{key_suffix}")

        with tabs[0]:
            _render_trades(df_all, "all")

        for i, pid in enumerate(STRATEGY_PORTFOLIOS, start=1):
            with tabs[i]:
                _render_trades(df_all[df_all["strategy"] == pid].copy(), pid)

# ---------------------------------------------------------------------------
# LIVE SIGNALS PAGE
# ---------------------------------------------------------------------------

elif page == "Live Signals":
    st.title("Live Signals")
    st.caption("Refreshes every ~5 minutes from Reddit, news, options flow, Google Trends, and macro crawlers")

    with st.spinner("Loading signals..."):
        context = get_signals()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Meme Signals", "📰 News Signals", "📊 Options Flow",
        "📈 Google Trends", "🌍 Macro",
    ])

    with tab1:
        st.subheader("Reddit / Social Momentum")
        meme = context.get("meme_signals", {})
        if meme:
            rows = []
            for sym, data in sorted(meme.items(), key=lambda x: x[1].get("mention_spike", 0), reverse=True):
                sentiment_emoji = "🟢" if data["sentiment"] > 0.15 else ("🔴" if data["sentiment"] < -0.1 else "🟡")
                rows.append({
                    "Symbol": sym,
                    "Mentions": data["mention_count"],
                    "Spike": f"{data['mention_spike']:.1f}×",
                    "Sentiment": f"{sentiment_emoji} {data['sentiment']:+.2f}",
                    "Sources": ", ".join(data["sources"]),
                    "Top Post": data["top_post"][:80] + "..." if len(data.get("top_post", "")) > 80 else data.get("top_post", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No meme signals — configure Reddit API or wait for next refresh.")

    with tab2:
        st.subheader("News & Earnings Signals")
        news = context.get("news_signals", {})
        if news:
            for sym, data in list(news.items())[:10]:
                with st.expander(f"**{sym}** — {len(data.get('articles', []))} articles"):
                    cols = st.columns(3)
                    cols[0].metric("Earnings Beat",
                                   "✅ Yes" if data.get("earnings_beat") is True
                                   else ("❌ Miss" if data.get("earnings_beat") is False else "—"))
                    cols[1].metric("Surprise %",
                                   f"{data.get('earnings_surprise_pct', 0) or 0:+.1f}%")
                    cols[2].metric("Insider Buys", data.get("insider_buys", 0))

                    for art in data.get("articles", [])[:3]:
                        st.write(f"• **{art.get('source', '')}**: {art.get('title', '')}")
        else:
            st.info("No news signals yet.")

    with tab3:
        st.subheader("Unusual Options Activity")
        opts = context.get("options_flow", {})
        if opts:
            rows = []
            for sym, data in opts.items():
                rows.append({
                    "Symbol": sym,
                    "Direction": data.get("direction", "").upper(),
                    "C/P Ratio": f"{data.get('call_put_ratio', 0):.1f}",
                    "Premium Ratio": f"{data.get('premium_ratio', 0):.1f}×",
                    "OI Change": f"{data.get('oi_change', 0):+.0%}",
                    "Block Trades": data.get("block_trade_count", 0),
                    "Avg DTE": data.get("avg_days_to_expiry", 0),
                    "Note": data.get("notable_trade", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No options flow data.")

    with tab4:
        st.subheader("Google Trends Search Interest")
        st.caption("Retail attention signal via pytrends (3-month window, US). Updates hourly.")
        trends = context.get("google_trends", {})
        if trends:
            rows = []
            for sym, data in sorted(trends.items(), key=lambda x: x[1].get("spike_ratio", 0), reverse=True):
                trend_icon = {"rising": "↑", "falling": "↓", "flat": "→"}.get(data.get("trend", "flat"), "→")
                rows.append({
                    "Symbol":        sym,
                    "Interest (0-100)": data.get("interest_score", 0),
                    "vs 3-month avg":   f"{data.get('spike_ratio', 1.0):.1f}×",
                    "Trend":         f"{trend_icon} {data.get('trend', 'flat').capitalize()}",
                    "Peak Week":     data.get("peak_week", "—"),
                })
            df_trends = pd.DataFrame(rows)
            st.dataframe(df_trends, use_container_width=True, hide_index=True)

            # Bar chart of spike ratios
            rising = {s: d for s, d in trends.items() if d.get("trend") == "rising"}
            if rising:
                st.markdown("**Rising-trend tickers**")
                fig_t = px.bar(
                    x=list(rising.keys()),
                    y=[v.get("spike_ratio", 1.0) for v in rising.values()],
                    labels={"x": "Ticker", "y": "Search spike vs 3-month avg"},
                    template="plotly_dark",
                )
                fig_t.add_hline(y=2.0, line_dash="dash", line_color="orange",
                                annotation_text="Meme signal threshold")
                st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("No Google Trends data yet — check pytrends is installed and retry.")

    with tab5:
        st.subheader("Macro Environment")
        st.caption("FRED (ECB rate, DE Bund yield, VIX, euro CPI, DE unemployment) + yfinance (DAX, VSTOXX) + CNN Fear & Greed. Updates hourly.")
        macro = context.get("macro", {})
        if macro:
            # Top KPI row
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Regime", macro.get("regime", "—").upper())
            fg_score = macro.get("fear_greed_score", 50)
            fg_label = macro.get("fear_greed_label", "Neutral")
            fg_delta  = "Extreme Fear" if fg_score < 25 else ("Fear" if fg_score < 40 else ("Neutral" if fg_score < 60 else ("Greed" if fg_score < 75 else "Extreme Greed")))
            c2.metric("Fear & Greed", f"{fg_score:.0f} / 100", fg_delta)
            c3.metric("VSTOXX", f"{macro.get('vstoxx', '—')}")
            c4.metric("VIX (global)", f"{macro.get('vix', '—')}")
            c5.metric("ECB Rate", f"{macro.get('ecb_rate', '—')}%")

            st.divider()
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("DAX 40", f"{macro.get('dax_level', '—'):,.0f}" if macro.get('dax_level') else "—")
            col_b.metric("DE Bund 10Y", f"{macro.get('de_yield_10y', '—')}%")
            col_c.metric("Euro CPI (index)", macro.get("cpi", "—"))
            col_d.metric("DE Unemployment", f"{macro.get('unemployment', '—')}%")

            # Fear & Greed gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fg_score,
                title={"text": f"Fear & Greed — {fg_label}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "white"},
                    "steps": [
                        {"range": [0, 25],  "color": "#d32f2f"},
                        {"range": [25, 45], "color": "#f57c00"},
                        {"range": [45, 55], "color": "#fbc02d"},
                        {"range": [55, 75], "color": "#388e3c"},
                        {"range": [75, 100],"color": "#1b5e20"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": fg_score,
                    },
                },
            ))
            gauge.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(gauge, use_container_width=True)

            st.caption(f"Source: {macro.get('source', '—')} | Last updated: {macro.get('timestamp', '—')}")
        else:
            st.info("No macro data yet — FRED + CNN Fear & Greed not fetched. Retry in a moment.")

# ---------------------------------------------------------------------------
# MARKET DATA PAGE
# ---------------------------------------------------------------------------

elif page == "Market Data":
    st.title("Market Data & Correlations")

    from config.universe import DAX40, GERMAN_ETFS, FULL_UNIVERSE

    mdb   = get_market_db()
    stats = get_market_stats()

    # DB health row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbols tracked",  stats.get("symbols", 0))
    c2.metric("Price bars stored", f"{stats.get('bars', 0):,}")
    c3.metric("Correlation pairs", f"{stats.get('correlations', 0):,}")
    c4.metric("Latest date",       stats.get("latest_date") or "—")

    if stats.get("bars", 0) == 0:
        st.warning(
            "No price data yet. Run:  `.venv/bin/python clawbot_cli.py update-prices`  "
            "then  `correlations --run`  to populate the database."
        )
        st.stop()

    tab_price, tab_heatmap, tab_pairs, tab_clusters = st.tabs([
        "Price History", "Correlation Heatmap", "Top Pairs", "Clusters"
    ])

    # ---- Price History ----
    with tab_price:
        available = mdb.get_available_symbols()
        sym = st.selectbox("Symbol", sorted(available), index=0 if available else 0)
        days_opt = st.select_slider("History (days)", [30, 60, 90, 180, 365], value=90)

        df_bars = get_price_bars_cached(sym, days_opt)
        if df_bars.empty:
            st.info(f"No data for {sym}.")
        else:
            # Candlestick chart
            fig_c = go.Figure(go.Candlestick(
                x=df_bars["date"],
                open=df_bars["open"], high=df_bars["high"],
                low=df_bars["low"],  close=df_bars["close"],
                name=sym,
            ))
            fig_c.update_layout(
                title=f"{sym} — {days_opt}d price history",
                xaxis_title="Date", yaxis_title="Price (€)",
                height=380, template="plotly_dark",
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig_c, use_container_width=True)

            # Volume bar chart
            fig_v = px.bar(df_bars, x="date", y="volume",
                           title=f"{sym} — Daily Volume",
                           template="plotly_dark",
                           color_discrete_sequence=["#4ecdc4"])
            fig_v.update_layout(height=200, margin=dict(t=30, b=0))
            st.plotly_chart(fig_v, use_container_width=True)

            # Top correlations for selected symbol
            st.subheader(f"Top Correlations for {sym}")
            period_sel = st.radio("Window", [30, 60, 90], index=1, horizontal=True, key="price_period")
            corrs = mdb.get_top_correlations(sym, period_days=period_sel, min_abs_r=0.4, limit=15)
            if corrs:
                df_c = pd.DataFrame(corrs)
                df_c["direction"] = df_c["pearson_r"].apply(lambda r: "Positive" if r > 0 else "Negative")
                fig_bar = px.bar(
                    df_c, x="peer", y="pearson_r", color="direction",
                    color_discrete_map={"Positive": "#2a9d8f", "Negative": "#e63946"},
                    title=f"{sym} correlations ({period_sel}d)",
                    template="plotly_dark",
                )
                fig_bar.add_hline(y=0, line_color="gray")
                st.plotly_chart(fig_bar, use_container_width=True)
                st.dataframe(df_c[["peer","pearson_r","lag_days","p_value"]],
                             use_container_width=True, hide_index=True)
            else:
                st.info("No significant correlations found. Run `correlations --run` first.")

    # ---- Correlation Heatmap ----
    with tab_heatmap:
        col_left, col_right = st.columns([3, 1])
        with col_right:
            period_h = st.radio("Window (days)", [30, 60, 90], index=1, key="heatmap_period")
            group    = st.selectbox("Symbol group", ["DAX 40", "German ETFs", "Trading Universe"])

        if group == "DAX 40":
            hm_symbols = [s for s in DAX40 if s in mdb.get_available_symbols()][:20]
        elif group == "German ETFs":
            hm_symbols = [s for s in GERMAN_ETFS if s in mdb.get_available_symbols()]
        else:
            hm_symbols = [s for s in STRATEGY_PORTFOLIOS or [] if s in mdb.get_available_symbols()]
            if not hm_symbols:
                hm_symbols = [s for s in mdb.get_available_symbols()][:25]

        hm_symbols = hm_symbols[:25]  # cap for readability
        symbols_key = ",".join(sorted(hm_symbols))

        with col_left:
            if len(hm_symbols) < 2:
                st.info("Not enough symbols with data yet.")
            else:
                corr_df = get_correlation_matrix_cached(symbols_key, period_h)
                fig_hm = px.imshow(
                    corr_df,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title=f"Correlation Matrix ({period_h}d)",
                    template="plotly_dark",
                    aspect="auto",
                )
                fig_hm.update_layout(height=600)
                st.plotly_chart(fig_hm, use_container_width=True)

    # ---- Top Pairs ----
    with tab_pairs:
        c_period, c_minr = st.columns(2)
        pair_period = c_period.radio("Window (days)", [30, 60, 90], index=1, key="pairs_period")
        min_r_sel   = c_minr.slider("Min |r|", 0.4, 0.95, 0.65, 0.05)

        pairs = get_top_pairs_cached(pair_period, min_r_sel)
        if pairs:
            df_pairs = pd.DataFrame(pairs)
            df_pairs["direction"] = df_pairs["pearson_r"].apply(
                lambda r: "Positive" if r > 0 else "Negative"
            )
            df_pairs["abs_r"] = df_pairs["pearson_r"].abs()
            st.metric("Pairs found", len(df_pairs))
            st.dataframe(
                df_pairs.sort_values("abs_r", ascending=False)
                        [["symbol_a","symbol_b","pearson_r","p_value","direction"]],
                use_container_width=True, hide_index=True,
            )

            # Scatter: distribution of r values
            fig_dist = px.histogram(
                df_pairs, x="pearson_r", nbins=30, color="direction",
                color_discrete_map={"Positive": "#2a9d8f", "Negative": "#e63946"},
                title="Distribution of correlation coefficients",
                template="plotly_dark",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No pairs yet. Run `correlations --run` to compute.")

    # ---- Clusters ----
    with tab_clusters:
        st.caption("Stocks that all move together (r ≥ 0.70 within the cluster).")
        from core.correlation_engine import CorrelationEngine

        @st.cache_data(ttl=3600)
        def _get_clusters():
            return CorrelationEngine(get_market_db()).find_clusters(threshold=0.70)

        clusters = _get_clusters()
        if clusters:
            for i, cl in enumerate(clusters[:10], 1):
                with st.expander(
                    f"Cluster {i}: {' · '.join(cl['symbols'][:6])}{'…' if len(cl['symbols']) > 6 else ''}"
                    f"  (avg r = {cl['avg_r']:.2f})",
                    expanded=(i == 1),
                ):
                    st.write("**Members:** " + ", ".join(cl["symbols"]))
                    st.write(f"**Size:** {cl['size']} | **Avg intra-cluster r:** {cl['avg_r']}")
        else:
            st.info("No clusters found. Run `correlations --run` first.")

# ---------------------------------------------------------------------------
# RISK & SAFETY PAGE
# ---------------------------------------------------------------------------
# SELF-IMPROVEMENT PAGE
# ---------------------------------------------------------------------------

elif page == "Self-Improvement":
    st.title("Self-Improvement Engine")
    st.caption(
        "Runs every 6 hours after market close. Adjusts strategy parameters within "
        "hard bounds, rebalances capital allocation, and researches new insights."
    )

    opt_status = optimizer.get_status()
    current_params = optimizer.get_current_params()

    # Last run summary
    last_run = opt_status.get("last_run")
    st.metric("Last Optimization Run", last_run or "Never")

    tab_weights, tab_params, tab_research, tab_history = st.tabs([
        "Capital Weights", "Live Parameters", "Web Research Log", "Run History"
    ])

    # ---- Capital Weights ----
    with tab_weights:
        st.subheader("Strategy Allocation Weights")
        st.caption("Rebalanced based on rolling Sharpe ratio. Bounded 5%–40% per strategy.")
        weights = opt_status.get("weights", {})
        if weights:
            df_w = pd.DataFrame([
                {"Strategy": k.replace("_", " ").title(), "Weight": v, "Pct": f"{v:.1%}"}
                for k, v in weights.items()
            ])
            fig = px.pie(
                df_w, values="Weight", names="Strategy",
                title="Current Capital Allocation",
                template="plotly_dark",
                hole=0.35,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_w[["Strategy", "Pct"]], use_container_width=True, hide_index=True)
        else:
            st.info("No allocation data yet. Run the optimizer at least once.")

    # ---- Live Parameters ----
    with tab_params:
        st.subheader("Current Strategy Parameters")
        st.caption("These values are live — the optimizer adjusts them within the bounds shown.")

        for strategy_name, params in current_params.items():
            bounds = PARAM_BOUNDS.get(strategy_name, {})
            with st.expander(f"**{strategy_name.replace('_', ' ').title()}**", expanded=True):
                if not params:
                    st.write("No tunable parameters.")
                    continue

                rows = []
                for param, value in params.items():
                    lo, hi, step = bounds.get(param, (None, None, None))
                    # Cooling period info
                    last_changed = opt_status.get("cooling_periods", {}).get(strategy_name, {}).get(param)
                    rows.append({
                        "Parameter": param,
                        "Current Value": value,
                        "Min": lo,
                        "Max": hi,
                        "Step": step,
                        "Last Changed": last_changed or "Never",
                    })

                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---- Web Research Log ----
    with tab_research:
        st.subheader("Web Research Findings")
        from config.settings import TAVILY_API_KEY as _TAVILY_KEY
        _backend_label = "Tavily" if _TAVILY_KEY else "DuckDuckGo (set TAVILY_API_KEY to upgrade)"
        st.caption(f"Search backend: **{_backend_label}** — runs after each optimizer cycle.")
        findings = optimizer.get_research_log(limit=50)
        if findings:
            df_f = pd.DataFrame(findings)
            if "timestamp" in df_f.columns:
                df_f["timestamp"] = pd.to_datetime(df_f["timestamp"]).dt.strftime("%m-%d %H:%M")

            category_filter = st.selectbox(
                "Filter by category",
                ["All"] + list(df_f["category"].unique() if "category" in df_f.columns else [])
            )
            if category_filter != "All" and "category" in df_f.columns:
                df_f = df_f[df_f["category"] == category_filter]

            display_cols = [c for c in ["timestamp", "category", "ticker", "relevance_score", "snippet"]
                            if c in df_f.columns]
            st.dataframe(df_f[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No research log yet. The optimizer will populate this after its first run.")

    # ---- Run History ----
    with tab_history:
        st.subheader("Optimization Run History")
        history = opt_status.get("run_history", [])
        if history:
            df_h = pd.DataFrame(history)
            if "timestamp" in df_h.columns:
                df_h["timestamp"] = pd.to_datetime(df_h["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

            st.dataframe(df_h, use_container_width=True, hide_index=True)

            # Market regime chart
            if "market_regime" in df_h.columns:
                regime_counts = df_h["market_regime"].value_counts().reset_index()
                regime_counts.columns = ["Regime", "Count"]
                fig_r = px.bar(regime_counts, x="Regime", y="Count",
                               title="Detected Market Regimes",
                               template="plotly_dark",
                               color="Regime",
                               color_discrete_map={
                                   "bull": "#2a9d8f", "bear": "#e63946",
                                   "volatile": "#f4a261", "unknown": "#adb5bd"
                               })
                st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("No optimization runs recorded yet.")

# ---------------------------------------------------------------------------

elif page == "Risk & Safety":
    import re as _re
    from pathlib import Path as _Path
    from config import safety_constants as SC
    from core.safety import seal_safety_constants as _seal

    def _apply_safety_constants(updates: dict) -> str:
        """Rewrite specific constants in safety_constants.py, reseal, return new hash."""
        # Reload the module so we always read the current file
        import importlib, config.safety_constants as _sc_mod
        sc_path = _Path(__file__).parent.parent / "config" / "safety_constants.py"
        content = sc_path.read_text()
        for name, value in updates.items():
            if isinstance(value, bool):
                pattern     = rf'^({_re.escape(name)}\s*=\s*)(True|False)'
                replacement = rf'\g<1>{value}'
            elif isinstance(value, int):
                pattern     = rf'^({_re.escape(name)}\s*=\s*)\d+'
                replacement = rf'\g<1>{value}'
            else:   # float
                pattern     = rf'^({_re.escape(name)}\s*=\s*)[\d.]+'
                replacement = rf'\g<1>{round(value, 4)}'
            content = _re.sub(pattern, replacement, content, flags=_re.MULTILINE)
        sc_path.write_text(content)
        new_hash = _seal()
        importlib.reload(_sc_mod)
        return new_hash

    st.title("Risk & Safety Status")

    # ── Circuit Breaker ──────────────────────────────────────────────
    st.subheader("Circuit Breaker")
    cb = safety.get_state()
    level_labels = {
        "NORMAL":  "🟢 NORMAL",
        "LEVEL_1": "🟡 LEVEL 1 — Sizes halved",
        "LEVEL_2": "🔴 LEVEL 2 — No new orders",
        "LEVEL_3": "🚨 LEVEL 3 — FULL HALT",
    }
    st.metric("Status", level_labels.get(cb.level.value, cb.level.value))
    if cb.triggered_at:
        st.write(f"**Triggered at:** {cb.triggered_at}")
        st.write(f"**Reason:** {cb.triggered_reason}")

    # ── Current values ───────────────────────────────────────────────
    st.divider()
    st.subheader("Safety Configuration")
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Position Size",    f"{SC.MAX_POSITION_FRACTION:.0%}")
    c1.metric("Max Meme Position",    f"{SC.MAX_MEME_POSITION_FRACTION:.0%}")
    c1.metric("Max Meme Allocation",  f"{SC.MAX_MEME_ALLOCATION:.0%}")
    c2.metric("L1 Daily Loss",        f"{SC.CIRCUIT_BREAKER_L1_DAILY_LOSS:.0%}")
    c2.metric("L2 Daily Loss",        f"{SC.CIRCUIT_BREAKER_L2_DAILY_LOSS:.0%}")
    c2.metric("L3 Max Drawdown",      f"{SC.CIRCUIT_BREAKER_L3_DRAWDOWN:.0%}")
    c3.metric("Max Orders/Min",       SC.MAX_ORDERS_PER_MINUTE)
    c3.metric("Max Orders/Day",       SC.MAX_ORDERS_PER_DAY)
    c3.metric("Min Stock Price",      f"€{SC.MIN_STOCK_PRICE:.2f}")

    flags_display = {
        "Leverage":        f"{SC.MAX_LEVERAGE}× (none)",
        "Short Selling":   "✅ ON" if SC.SHORT_SELLING   else "❌ OFF",
        "Options Writing": "✅ ON" if SC.OPTIONS_WRITING else "❌ OFF",
        "Futures":         "✅ ON" if SC.FUTURES_TRADING else "❌ OFF",
        "Crypto":          "✅ ON" if SC.CRYPTO_TRADING  else "❌ OFF",
        "Live Trading":    "⚠️ ENABLED" if __import__(
            'config.settings', fromlist=['LIVE_TRADING_ENABLED']
        ).LIVE_TRADING_ENABLED else "❌ DISABLED",
        "Audit Log":       "✅ Append-Only",
    }
    st.dataframe(
        pd.DataFrame(list(flags_display.items()), columns=["Setting", "Value"]),
        use_container_width=True, hide_index=True,
    )

    # ── Edit form ────────────────────────────────────────────────────
    st.divider()
    with st.expander("Edit Safety Constants", expanded=False):
        st.warning(
            "Changes rewrite `config/safety_constants.py` and regenerate the "
            "integrity hash. The engine picks up the new values on its next cycle."
        )
        with st.form("safety_edit_form"):
            st.subheader("Position Limits")
            e1, e2 = st.columns(2)
            new_max_pos      = e1.number_input(
                "Max Position Size (%)", min_value=1, max_value=20,
                value=int(round(SC.MAX_POSITION_FRACTION * 100)), step=1,
            )
            new_max_strat    = e2.number_input(
                "Max Strategy Allocation (%)", min_value=10, max_value=50,
                value=int(round(SC.MAX_STRATEGY_ALLOCATION * 100)), step=5,
            )
            e3, e4 = st.columns(2)
            new_max_meme     = e3.number_input(
                "Max Meme Allocation (%)", min_value=1, max_value=25,
                value=int(round(SC.MAX_MEME_ALLOCATION * 100)), step=1,
            )
            new_max_meme_pos = e4.number_input(
                "Max Meme Position Size (%)", min_value=1, max_value=10,
                value=int(round(SC.MAX_MEME_POSITION_FRACTION * 100)), step=1,
            )
            new_min_price    = st.number_input(
                "Min Stock Price (€)", min_value=0.10, max_value=10.0,
                value=float(SC.MIN_STOCK_PRICE), step=0.10, format="%.2f",
            )

            st.subheader("Order Rate Limits")
            e5, e6 = st.columns(2)
            new_orders_min   = e5.number_input(
                "Max Orders / Minute", min_value=1, max_value=20,
                value=int(SC.MAX_ORDERS_PER_MINUTE), step=1,
            )
            new_orders_day   = e6.number_input(
                "Max Orders / Day", min_value=10, max_value=500,
                value=int(SC.MAX_ORDERS_PER_DAY), step=10,
            )

            st.subheader("Feature Flags")
            f1, f2, f3, f4 = st.columns(4)
            new_short   = f1.checkbox("Short Selling",   value=bool(SC.SHORT_SELLING))
            new_options = f2.checkbox("Options Writing", value=bool(SC.OPTIONS_WRITING))
            new_futures = f3.checkbox("Futures Trading", value=bool(SC.FUTURES_TRADING))
            new_crypto  = f4.checkbox("Crypto Trading",  value=bool(SC.CRYPTO_TRADING))

            save_btn = st.form_submit_button("Save & Reseal", type="primary")

        if save_btn:
            try:
                new_hash = _apply_safety_constants({
                    "MAX_POSITION_FRACTION":    new_max_pos  / 100,
                    "MAX_STRATEGY_ALLOCATION":  new_max_strat / 100,
                    "MAX_MEME_ALLOCATION":       new_max_meme / 100,
                    "MAX_MEME_POSITION_FRACTION": new_max_meme_pos / 100,
                    "MIN_STOCK_PRICE":           new_min_price,
                    "MAX_ORDERS_PER_MINUTE":     int(new_orders_min),
                    "MAX_ORDERS_PER_DAY":        int(new_orders_day),
                    "SHORT_SELLING":             new_short,
                    "OPTIONS_WRITING":           new_options,
                    "FUTURES_TRADING":           new_futures,
                    "CRYPTO_TRADING":            new_crypto,
                })
                st.success(f"Saved and resealed. New hash: `{new_hash}`")
                st.rerun()
            except Exception as _err:
                st.error(f"Failed to save: {_err}")

    # ── Audit log ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Today's Audit Log")
    records = read_audit_log()
    if records:
        df_audit = pd.DataFrame(records)
        cols_to_show = [c for c in ["_ts", "event", "level", "symbol", "action", "message", "reason", "approved"]
                        if c in df_audit.columns]
        st.dataframe(df_audit[cols_to_show].tail(50), use_container_width=True, hide_index=True)
    else:
        st.info("No audit records today yet.")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()
