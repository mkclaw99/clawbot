"""
Append-only audit logger.
The bot can WRITE but never DELETE or OVERWRITE audit records.
Every order, signal, parameter change, and safety event is recorded here.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from config.safety_constants import AUDIT_LOG_DIR, AUDIT_LOG_APPEND_ONLY

_AUDIT_DIR = Path(AUDIT_LOG_DIR)
_AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def _audit_file() -> Path:
    """One log file per UTC day."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _AUDIT_DIR / f"audit_{today}.jsonl"


def _write(record: dict) -> None:
    """Write a single JSON record to today's append-only audit file."""
    if not AUDIT_LOG_APPEND_ONLY:
        raise RuntimeError("AUDIT_LOG_APPEND_ONLY has been tampered with — refusing to write.")

    record["_ts"] = datetime.now(timezone.utc).isoformat()
    path = _audit_file()

    # Open in append mode — never truncate
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Public audit functions — call these from anywhere in the system
# ---------------------------------------------------------------------------

def audit_order(action: str, symbol: str, qty: float, price: float,
                strategy: str, reason: str, approved: bool, **extra: Any) -> None:
    _write({
        "event": "ORDER",
        "action": action,       # BUY | SELL | BLOCKED
        "symbol": symbol,
        "qty": qty,
        "price": price,
        "strategy": strategy,
        "reason": reason,
        "approved": approved,
        **extra,
    })
    logger.info(f"[AUDIT ORDER] {action} {qty}x {symbol} @ {price:.2f} ({strategy}) approved={approved}")


def audit_safety_event(level: str, message: str, **extra: Any) -> None:
    _write({
        "event": "SAFETY",
        "level": level,         # INFO | WARN | CIRCUIT_BREAKER | HALT
        "message": message,
        **extra,
    })
    logger.warning(f"[AUDIT SAFETY] [{level}] {message}")


def audit_signal(strategy: str, symbol: str, signal: str,
                 score: float, source: str, **extra: Any) -> None:
    _write({
        "event": "SIGNAL",
        "strategy": strategy,
        "symbol": symbol,
        "signal": signal,       # BUY | SELL | HOLD
        "score": score,
        "source": source,
        **extra,
    })


def audit_param_change(component: str, param: str,
                       old_val: Any, new_val: Any, reason: str) -> None:
    _write({
        "event": "PARAM_CHANGE",
        "component": component,
        "param": param,
        "old": old_val,
        "new": new_val,
        "reason": reason,
    })
    logger.info(f"[AUDIT PARAM] {component}.{param}: {old_val} → {new_val}")


def audit_system(event: str, message: str, **extra: Any) -> None:
    _write({
        "event": event,
        "message": message,
        **extra,
    })


def read_audit_log(
    date: str | None = None,
    event_types: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """
    Read audit records for a given date (YYYY-MM-DD), defaulting to today.

    Args:
        date:        UTC date string (YYYY-MM-DD); defaults to today.
        event_types: If given, only return records whose 'event' field is in this list.
        limit:       If given, return only the last N records (after filtering).
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = _AUDIT_DIR / f"audit_{date}.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if event_types:
        records = [r for r in records if r.get("event") in event_types]
    if limit is not None:
        records = records[-limit:]
    return records
