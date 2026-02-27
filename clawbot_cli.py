"""
ClawBot CLI — OpenClaw skill interface
========================================
Exposes all ClawBot operations as JSON-output subcommands.
OpenClaw skills invoke these via bash tool calls.

Usage:
    .venv/bin/python clawbot_cli.py <subcommand> [args]

All subcommands print JSON to stdout and exit 0 on success, 1 on error.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on PYTHONPATH when invoked from anywhere
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
logger.remove()                              # suppress to stderr only, no noise on stdout
logger.add(sys.stderr, level="WARNING")


def _out(data: dict | list) -> None:
    """Print JSON to stdout and exit 0."""
    print(json.dumps(data, default=str, indent=2))
    sys.exit(0)


def _err(msg: str, **extra) -> None:
    """Print error JSON to stdout and exit 1."""
    print(json.dumps({"error": msg, **extra}, default=str, indent=2))
    sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

STRATEGY_PORTFOLIOS = [
    "meme_momentum",
    "technical_trend",
    "mean_reversion",
    "options_flow",
    "macro_news",
]


def _portfolio_summary(portfolio_id: str) -> dict:
    """Return a summary dict for one strategy portfolio."""
    from core.broker import Broker
    from core.portfolio import PortfolioManager

    broker = Broker(portfolio_id=portfolio_id)
    pm     = PortfolioManager(broker)
    state  = pm.sync()
    return {
        "strategy":       portfolio_id,
        "portfolio_value": round(state.portfolio_value, 2),
        "cash":            round(state.cash, 2),
        "equity":          round(state.equity, 2),
        "total_pnl":       round(state.total_pnl, 2),
        "total_pnl_pct":   round(state.total_pnl / state.start_value * 100, 3) if state.start_value else 0,
        "position_count":  len(state.positions),
        "positions": [
            {
                "symbol":            p.symbol,
                "qty":               p.qty,
                "avg_entry_price":   p.avg_entry_price,
                "current_price":     p.current_price,
                "market_value":      p.market_value,
                "unrealized_pnl":    p.unrealized_pnl,
                "unrealized_pnl_pct": p.unrealized_pnl_pct,
            }
            for p in state.positions
        ],
    }


def cmd_portfolio(_args) -> None:
    """Show all strategy portfolios side-by-side for comparison."""
    portfolios = [_portfolio_summary(pid) for pid in STRATEGY_PORTFOLIOS]

    combined_value = sum(p["portfolio_value"] for p in portfolios)
    combined_pnl   = sum(p["total_pnl"] for p in portfolios)

    _out({
        "portfolios":      portfolios,
        "combined_value":  round(combined_value, 2),
        "combined_pnl":    round(combined_pnl, 2),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    })


def cmd_safety_status(_args) -> None:
    """Circuit breaker level and recent safety events."""
    from core.safety import SafetyLayer
    from core.audit import read_audit_log

    safety = SafetyLayer()
    state  = safety.get_state()

    recent = []
    try:
        entries = read_audit_log(event_types=["SAFETY"], limit=10)
        recent  = entries
    except Exception:
        pass

    _out({
        "level":               state.level.value,
        "triggered_at":        state.triggered_at,
        "triggered_reason":    state.triggered_reason,
        "daily_start_value":   state.daily_start_value,
        "peak_portfolio_value": state.peak_portfolio_value,
        "size_multiplier":     safety.get_size_multiplier(),
        "is_halted":           safety.is_halted(),
        "recent_safety_events": recent,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    })


def cmd_macro(_args) -> None:
    """Current macro context: VIX, yield spread, Fear & Greed, regime."""
    from crawler.macro_crawler import fetch_macro_context

    ctx = fetch_macro_context()
    _out(ctx)


def cmd_scan(_args) -> None:
    """
    Run all crawlers + all 5 strategies.
    Returns signals without executing anything.
    """
    from core.broker import Broker
    from core.portfolio import PortfolioManager
    from core.safety import SafetyLayer
    from crawler.signal_aggregator import SignalAggregator
    from strategies.meme_momentum import MemeMomentumStrategy
    from strategies.technical_trend import TechnicalTrendStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.options_flow import OptionsFlowStrategy
    from strategies.macro_news import MacroNewsStrategy

    # Aggregate held positions across all strategy portfolios
    held_set: set[str] = set()
    combined_value = 0.0
    for pid in STRATEGY_PORTFOLIOS:
        broker = Broker(portfolio_id=pid)
        pm     = PortfolioManager(broker)
        state  = pm.sync()
        held_set     |= {p.symbol for p in state.positions}
        combined_value += state.portfolio_value

    aggregator = SignalAggregator()
    context    = aggregator.refresh(force=True)
    context["current_positions"] = held_set

    strategies = [
        MemeMomentumStrategy(),
        TechnicalTrendStrategy(),
        MeanReversionStrategy(),
        OptionsFlowStrategy(),
        MacroNewsStrategy(),
    ]

    all_signals = []
    for strategy in strategies:
        try:
            sigs = strategy.generate_signals(context.get("universe", []), context)
            all_signals.extend(sigs)
        except Exception as e:
            logger.warning(f"Strategy {strategy.name} error: {e}")

    # Deduplicate: keep highest-confidence per symbol
    sig_map: dict = {}
    for sig in all_signals:
        if sig.symbol not in sig_map or sig.confidence > sig_map[sig.symbol].confidence:
            sig_map[sig.symbol] = sig

    signals_out = [
        {
            "symbol":     sig.symbol,
            "action":     sig.action,
            "score":      sig.score,
            "confidence": sig.confidence,
            "strategy":   sig.strategy,
            "reason":     sig.reason,
            "is_meme":    sig.is_meme,
            "is_actionable": sig.is_actionable,
        }
        for sig in sorted(sig_map.values(), key=lambda s: s.confidence, reverse=True)
    ]

    _out({
        "signals":         signals_out,
        "signal_count":    len(signals_out),
        "portfolio_value": combined_value,
        "macro_regime":    context.get("macro", {}).get("regime", "unknown"),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    })


def cmd_trade(args) -> None:
    """
    Execute a specific trade through the SafetyLayer.
    Args: --symbol, --action BUY|SELL, --reason, [--qty N], [--portfolio STRATEGY]
    """
    from core.broker import Broker
    from core.portfolio import PortfolioManager
    from core.safety import SafetyLayer

    strategy  = getattr(args, "portfolio", None) or "manual"
    broker    = Broker(portfolio_id=strategy)
    safety    = SafetyLayer()
    portfolio = PortfolioManager(broker)
    state     = portfolio.sync()

    symbol   = args.symbol.upper()
    action   = args.action.upper()
    reason   = args.reason or "manual"

    if action not in ("BUY", "SELL"):
        _err(f"action must be BUY or SELL, got: {action}")

    portfolio_value   = state.portfolio_value
    current_pos_val   = portfolio.get_position_value(symbol)
    strategy_exposure = portfolio.get_strategy_exposure(strategy)
    size_mult         = safety.get_size_multiplier()

    if safety.is_halted():
        _err("Safety halt is active — trading blocked", level=safety.get_state().level.value)

    if action == "BUY":
        price = broker.get_latest_price(symbol)
        if price <= 0:
            _err(f"Could not get price for {symbol}")

        qty = args.qty if args.qty else max(1, int(portfolio_value * 0.02 * size_mult / price))

        allowed, rej_reason = safety.check_order(
            symbol=symbol, action="BUY", qty=qty, price=price,
            strategy=strategy, portfolio_value=portfolio_value,
            current_position_value=current_pos_val,
            strategy_exposure=strategy_exposure,
            is_meme=False,
        )
        if not allowed:
            _out({"allowed": False, "rejection_reason": rej_reason,
                  "symbol": symbol, "action": action})

        result = broker.market_buy(symbol, qty)
        portfolio.record_trade(result.order_id, symbol, "BUY",
                               qty, result.filled_price or price, strategy, False)
        _out({
            "allowed":      True,
            "order_id":     result.order_id,
            "symbol":       symbol,
            "action":       "BUY",
            "qty":          qty,
            "filled_price": result.filled_price,
            "status":       result.status,
            "reason":       reason,
        })

    else:  # SELL
        pos = broker.get_position(symbol)
        if not pos or pos.qty <= 0:
            _err(f"No open position in {symbol}")

        price    = pos.current_price
        sell_qty = args.qty if args.qty else pos.qty

        allowed, rej_reason = safety.check_order(
            symbol=symbol, action="SELL", qty=sell_qty, price=price,
            strategy=strategy, portfolio_value=portfolio_value,
            current_position_value=current_pos_val,
            strategy_exposure=strategy_exposure,
            is_meme=False,
        )
        if not allowed:
            _out({"allowed": False, "rejection_reason": rej_reason,
                  "symbol": symbol, "action": action})

        result = broker.market_sell(symbol, sell_qty)
        portfolio.record_trade(result.order_id, symbol, "SELL",
                               sell_qty, result.filled_price or price, strategy, False)
        _out({
            "allowed":      True,
            "order_id":     result.order_id,
            "symbol":       symbol,
            "action":       "SELL",
            "qty":          sell_qty,
            "filled_price": result.filled_price,
            "status":       result.status,
            "reason":       reason,
        })


def cmd_optimize(_args) -> None:
    """Run the self-improvement optimizer (after-hours only)."""
    from core.optimizer import Optimizer
    from crawler.signal_aggregator import SignalAggregator
    from strategies.meme_momentum import MemeMomentumStrategy
    from strategies.technical_trend import TechnicalTrendStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.options_flow import OptionsFlowStrategy
    from strategies.macro_news import MacroNewsStrategy

    strategies = [
        MemeMomentumStrategy(),
        TechnicalTrendStrategy(),
        MeanReversionStrategy(),
        OptionsFlowStrategy(),
        MacroNewsStrategy(),
    ]
    aggregator = SignalAggregator()
    optimizer  = Optimizer(strategies, aggregator)
    summary    = optimizer.run()
    _out(summary)


def cmd_audit(args) -> None:
    """Return the last N audit log entries."""
    from core.audit import read_audit_log

    tail = args.tail if args.tail else 20
    entries = read_audit_log(limit=tail)
    _out({"entries": entries, "count": len(entries)})


def cmd_update_prices(args) -> None:
    """
    Download historical price data into market.db.
    --full  : 2 years of data (bootstrap)
    default : 1 year of data (nightly refresh)
    """
    from crawler.price_historian import PriceHistorian
    period = "2y" if getattr(args, "full", False) else "1y"
    historian = PriceHistorian()
    summary   = historian.update(period=period)
    _out(summary)


def cmd_correlations(args) -> None:
    """
    Run (or view) nightly correlation analysis.
    --run   : recompute correlations now and save to DB
    --symbol TICKER : show top correlated peers for a specific symbol
    --clusters : show correlation clusters
    --pairs : show all high-correlation pairs
    """
    from core.market_db import MarketDatabase
    from core.correlation_engine import CorrelationEngine

    db     = MarketDatabase()
    engine = CorrelationEngine(db)

    if getattr(args, "run", False):
        summary = engine.run()
        _out(summary)

    elif getattr(args, "symbol", None):
        sym  = args.symbol.upper()
        corr = db.get_top_correlations(sym, period_days=60, min_abs_r=0.4, limit=15)
        _out({
            "symbol":          sym,
            "period_days":     60,
            "top_correlations": corr,
            "computed_at":     db.get_latest_computed_at(),
        })

    elif getattr(args, "clusters", False):
        clusters = engine.find_clusters(period_days=60, threshold=0.70)
        _out({"clusters": clusters, "count": len(clusters)})

    else:
        # Default: show top pairs + DB stats
        pairs = db.get_all_correlation_pairs(period_days=60, min_abs_r=0.70)
        stats = db.get_stats()
        _out({
            "db_stats":       stats,
            "computed_at":    db.get_latest_computed_at(),
            "top_pairs":      pairs[:30],
        })


def cmd_search(args) -> None:
    """
    Run a web search query via Tavily (or DuckDuckGo fallback).
    Returns structured findings with ticker extraction and relevance scores.
    """
    from crawler.web_searcher import WebResearcher

    researcher = WebResearcher()
    findings = researcher.search(
        query=args.query,
        category=args.category,
        max_results=args.max_results,
    )
    _out({
        "query":    args.query,
        "backend":  researcher.backend,
        "count":    len(findings),
        "findings": [
            {
                "ticker":          f.ticker,
                "snippet":         f.snippet,
                "source_url":      f.source_url,
                "relevance_score": f.relevance_score,
                "category":        f.category,
            }
            for f in findings
        ],
    })


def cmd_run_cycle(_args) -> None:
    """
    Full trading cycle: portfolio sync → signal scan → execute qualifying signals
    → optimizer check (if after hours). Returns a summary of what happened.
    """
    import zoneinfo
    from core.broker import Broker
    from core.portfolio import PortfolioManager
    from core.safety import SafetyLayer, BreakerLevel
    from crawler.signal_aggregator import SignalAggregator
    from strategies.meme_momentum import MemeMomentumStrategy
    from strategies.technical_trend import TechnicalTrendStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.options_flow import OptionsFlowStrategy
    from strategies.macro_news import MacroNewsStrategy

    executed: list[dict] = []
    blocked:  list[dict] = []
    errors:   list[str]  = []

    # --- Per-strategy brokers + portfolio managers ---
    safety = SafetyLayer()
    brokers    = {pid: Broker(portfolio_id=pid)       for pid in STRATEGY_PORTFOLIOS}
    portfolios = {pid: PortfolioManager(brokers[pid]) for pid in STRATEGY_PORTFOLIOS}
    states     = {pid: portfolios[pid].sync()         for pid in STRATEGY_PORTFOLIOS}

    combined_value = sum(s.portfolio_value for s in states.values())
    safety.update_portfolio_value(combined_value)
    breaker_level = safety.get_state().level.value

    # --- Market hours check — XETRA (Frankfurt) 09:00–17:30 CET/CEST ---
    try:
        berlin = zoneinfo.ZoneInfo("Europe/Berlin")
        now = datetime.now(berlin)
        market_open  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
        market_close = now.replace(hour=17, minute=30, second=0, microsecond=0)
        is_market_hours = (
            now.weekday() < 5
            and market_open <= now <= market_close
        )
    except Exception:
        is_market_hours = True

    # --- Signal scan (each strategy scans its own held positions) ---
    aggregator = SignalAggregator()
    context    = aggregator.refresh(force=True)

    strategy_classes = [
        MemeMomentumStrategy(),
        TechnicalTrendStrategy(),
        MeanReversionStrategy(),
        OptionsFlowStrategy(),
        MacroNewsStrategy(),
    ]

    # Collect signals per-strategy (no deduplication — each strategy trades independently)
    strategy_signals: dict[str, list] = {}
    for strat in strategy_classes:
        pid = strat.name
        held = {p.symbol for p in states[pid].positions}
        ctx = {**context, "current_positions": held}
        try:
            sigs = strat.generate_signals(ctx.get("universe", []), ctx)
            strategy_signals[pid] = sigs
        except Exception as e:
            errors.append(f"{pid}: {e}")
            strategy_signals[pid] = []

    # Combined sig_map for top_signals display (best per symbol across all strategies)
    sig_map: dict = {}
    for sigs in strategy_signals.values():
        for sig in sigs:
            if sig.symbol not in sig_map or sig.confidence > sig_map[sig.symbol].confidence:
                sig_map[sig.symbol] = sig

    # --- Execute qualifying signals in each strategy's portfolio ---
    if is_market_hours and not safety.is_halted():
        size_mult = safety.get_size_multiplier()

        for strat_name, sigs in strategy_signals.items():
            broker    = brokers[strat_name]
            portfolio = portfolios[strat_name]
            state     = states[strat_name]

            for sig in sigs:
                if not sig.is_actionable or sig.confidence < 0.65:
                    continue
                symbol = sig.symbol
                try:
                    portfolio_value   = state.portfolio_value
                    current_pos_val   = portfolio.get_position_value(symbol)
                    strategy_exposure = portfolio.get_strategy_exposure(sig.strategy)

                    if sig.action == "BUY":
                        price = broker.get_latest_price(symbol)
                        if price <= 0:
                            continue
                        qty = max(1, int(portfolio_value * 0.02 * sig.confidence * size_mult / price))
                        allowed, reason = safety.check_order(
                            symbol=symbol, action="BUY", qty=qty, price=price,
                            strategy=sig.strategy, portfolio_value=portfolio_value,
                            current_position_value=current_pos_val,
                            strategy_exposure=strategy_exposure,
                            is_meme=sig.is_meme,
                        )
                        if not allowed:
                            blocked.append({"symbol": symbol, "action": "BUY",
                                            "strategy": strat_name, "reason": reason})
                            continue
                        result = broker.market_buy(symbol, qty)
                        portfolio.record_trade(result.order_id, symbol, "BUY",
                                               qty, result.filled_price or price,
                                               sig.strategy, sig.is_meme)
                        executed.append({
                            "symbol": symbol, "action": "BUY", "qty": qty,
                            "price": result.filled_price or price,
                            "strategy": strat_name, "confidence": sig.confidence,
                        })

                    elif sig.action == "SELL":
                        pos = broker.get_position(symbol)
                        if not pos or pos.qty <= 0:
                            continue
                        price    = pos.current_price
                        sell_qty = pos.qty
                        allowed, reason = safety.check_order(
                            symbol=symbol, action="SELL", qty=sell_qty, price=price,
                            strategy=sig.strategy, portfolio_value=portfolio_value,
                            current_position_value=current_pos_val,
                            strategy_exposure=strategy_exposure,
                            is_meme=sig.is_meme,
                        )
                        if not allowed:
                            blocked.append({"symbol": symbol, "action": "SELL",
                                            "strategy": strat_name, "reason": reason})
                            continue
                        result = broker.market_sell(symbol, sell_qty)
                        portfolio.record_trade(result.order_id, symbol, "SELL",
                                               sell_qty, result.filled_price or price,
                                               sig.strategy, sig.is_meme)
                        executed.append({
                            "symbol": symbol, "action": "SELL", "qty": sell_qty,
                            "price": result.filled_price or price,
                            "strategy": strat_name, "confidence": sig.confidence,
                        })

                except Exception as e:
                    errors.append(f"execute {strat_name}/{symbol}: {e}")

    # --- Nightly jobs (after hours only) ---
    optimizer_summary  = None
    price_update       = None
    correlation_update = None
    if not is_market_hours:
        # 1. Optimizer
        try:
            from core.optimizer import Optimizer
            opt = Optimizer(strategy_classes, aggregator)
            optimizer_summary = opt.run()
        except Exception as e:
            errors.append(f"optimizer: {e}")

        # 2. Price history update
        try:
            from crawler.price_historian import PriceHistorian
            price_update = PriceHistorian().update(period="1y")
        except Exception as e:
            errors.append(f"price-history: {e}")

        # 3. Correlation analysis (runs after prices are fresh)
        try:
            from core.market_db import MarketDatabase
            from core.correlation_engine import CorrelationEngine
            correlation_update = CorrelationEngine(MarketDatabase()).run()
        except Exception as e:
            errors.append(f"correlations: {e}")

    # Refresh all portfolios after trades
    final_states  = {pid: portfolios[pid].sync() for pid in STRATEGY_PORTFOLIOS}
    final_combined = sum(s.portfolio_value for s in final_states.values())
    final_pnl      = sum(s.total_pnl       for s in final_states.values())

    top_signals = [
        {
            "symbol":     sig.symbol,
            "action":     sig.action,
            "confidence": sig.confidence,
            "strategy":   sig.strategy,
            "reason":     sig.reason,
        }
        for sig in sorted(sig_map.values(), key=lambda s: s.confidence, reverse=True)[:5]
        if sig.is_actionable
    ]

    _out({
        "cycle_timestamp":   datetime.now(timezone.utc).isoformat(),
        "is_market_hours":   is_market_hours,
        "circuit_breaker":   breaker_level,
        "combined_value":    round(final_combined, 2),
        "combined_pnl":      round(final_pnl, 2),
        "portfolios": {
            pid: {
                "portfolio_value": round(s.portfolio_value, 2),
                "total_pnl":       round(s.total_pnl, 2),
                "position_count":  len(s.positions),
            }
            for pid, s in final_states.items()
        },
        "signals_generated": len(sig_map),
        "top_signals":       top_signals,
        "trades_executed":   executed,
        "trades_blocked":    blocked,
        "optimizer":          optimizer_summary,
        "price_update":       price_update,
        "correlation_update": correlation_update,
        "errors":             errors,
        "macro_regime":      context.get("macro", {}).get("regime", "unknown"),
    })


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clawbot",
        description="ClawBot CLI — JSON interface for OpenClaw skills",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("portfolio",     help="Current positions and P&L")
    sub.add_parser("safety-status", help="Circuit breaker state")
    sub.add_parser("macro",         help="Macro context (VIX, regime, Fear & Greed)")
    sub.add_parser("scan",          help="Run crawlers + strategies, return signals")
    sub.add_parser("optimize",      help="Run self-improvement optimizer")
    sub.add_parser("run-cycle",     help="Full cycle: scan + execute + optimize")

    p_trade = sub.add_parser("trade", help="Execute a specific trade")
    p_trade.add_argument("--symbol",    required=True,  help="Ticker symbol")
    p_trade.add_argument("--action",    required=True,  help="BUY or SELL")
    p_trade.add_argument("--reason",    default="manual", help="Reason for trade")
    p_trade.add_argument("--qty",       type=float, default=None, help="Quantity (optional)")
    p_trade.add_argument("--portfolio", default="manual",
                         help=f"Strategy portfolio to trade in ({', '.join(STRATEGY_PORTFOLIOS + ['manual'])})")

    p_audit = sub.add_parser("audit", help="Show recent audit log entries")
    p_audit.add_argument("--tail", type=int, default=20, help="Number of entries")

    p_search = sub.add_parser("search", help="Web search via Tavily (or DuckDuckGo fallback)")
    p_search.add_argument("--query",       required=True, help="Search query")
    p_search.add_argument("--category",    default="general",
                          choices=["general", "trending", "earnings", "regime", "strategy"],
                          help="Research category (default: general)")
    p_search.add_argument("--max-results", dest="max_results", type=int, default=5,
                          help="Max results to return (default: 5)")

    p_prices = sub.add_parser("update-prices",
                              help="Download historical OHLCV into market.db")
    p_prices.add_argument("--full", action="store_true",
                          help="Bootstrap 2 years of data (default: 1 year)")

    p_corr = sub.add_parser("correlations",
                             help="View or compute nightly correlation analysis")
    corr_group = p_corr.add_mutually_exclusive_group()
    corr_group.add_argument("--run",      action="store_true",
                             help="Recompute correlations now")
    corr_group.add_argument("--symbol",   metavar="TICKER",
                             help="Show top correlated peers for a symbol")
    corr_group.add_argument("--clusters", action="store_true",
                             help="Show correlation clusters (r ≥ 0.70)")
    corr_group.add_argument("--pairs",    action="store_true",
                             help="Show all high-correlation pairs")

    args = parser.parse_args()

    dispatch = {
        "portfolio":      cmd_portfolio,
        "safety-status":  cmd_safety_status,
        "macro":          cmd_macro,
        "scan":           cmd_scan,
        "trade":          cmd_trade,
        "optimize":       cmd_optimize,
        "audit":          cmd_audit,
        "run-cycle":      cmd_run_cycle,
        "search":         cmd_search,
        "update-prices":  cmd_update_prices,
        "correlations":   cmd_correlations,
    }

    try:
        dispatch[args.command](args)
    except Exception as e:
        _err(str(e))


if __name__ == "__main__":
    main()
