"""
Tests for core/audit.py — append-only audit logger.
All tests write to a temp directory so no real log files are touched.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
from unittest.mock import patch
from datetime import datetime, timezone

import core.audit as audit_module
from core.audit import (
    audit_order, audit_safety_event, audit_signal,
    audit_param_change, audit_system, read_audit_log,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def temp_audit_dir(tmp_path, monkeypatch):
    """Redirect all audit writes to a temp directory."""
    audit_dir = tmp_path / "audit_logs"
    audit_dir.mkdir()
    monkeypatch.setattr(audit_module, "_AUDIT_DIR", audit_dir)
    # Patch _audit_file to use the temp dir
    original_audit_file = audit_module._audit_file

    def patched_audit_file():
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return audit_dir / f"audit_{today}.jsonl"

    monkeypatch.setattr(audit_module, "_audit_file", patched_audit_file)
    yield audit_dir


# ---------------------------------------------------------------------------
# Write / append behaviour
# ---------------------------------------------------------------------------

class TestAuditWrite:

    def test_record_written_to_file(self):
        audit_system("TEST", "hello world")
        records = read_audit_log()
        assert any(r.get("message") == "hello world" for r in records)

    def test_record_has_timestamp(self):
        audit_system("TEST", "ts check")
        records = read_audit_log()
        assert all("_ts" in r for r in records)

    def test_multiple_writes_all_present(self):
        audit_system("TEST", "first")
        audit_system("TEST", "second")
        audit_system("TEST", "third")
        records = read_audit_log()
        messages = [r.get("message") for r in records]
        assert "first" in messages
        assert "second" in messages
        assert "third" in messages

    def test_file_is_valid_jsonl(self, temp_audit_dir):
        audit_system("TEST", "jsonl test")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = temp_audit_dir / f"audit_{today}.jsonl"
        for line in path.read_text().splitlines():
            json.loads(line)   # must not raise

    def test_records_appended_not_overwritten(self):
        audit_system("A", "first")
        audit_system("B", "second")
        records = read_audit_log()
        assert len(records) >= 2

    def test_refuses_to_write_when_append_only_disabled(self, monkeypatch):
        import config.safety_constants as SC
        monkeypatch.setattr(SC, "AUDIT_LOG_APPEND_ONLY", False)
        monkeypatch.setattr(audit_module, "AUDIT_LOG_APPEND_ONLY", False)
        with pytest.raises(RuntimeError, match="tampered"):
            audit_system("TEST", "should fail")


# ---------------------------------------------------------------------------
# audit_order
# ---------------------------------------------------------------------------

class TestAuditOrder:

    def test_order_fields_present(self):
        audit_order("BUY", "AAPL", 10, 150.0, "tech", "strong signal", True)
        records = read_audit_log(event_types=["ORDER"])
        assert len(records) >= 1
        r = records[-1]
        assert r["event"] == "ORDER"
        assert r["action"] == "BUY"
        assert r["symbol"] == "AAPL"
        assert r["qty"] == 10
        assert r["price"] == 150.0
        assert r["strategy"] == "tech"
        assert r["approved"] is True

    def test_blocked_order_recorded(self):
        audit_order("BUY", "TSLA", 5, 200.0, "meme", "position limit", False)
        records = read_audit_log(event_types=["ORDER"])
        blocked = [r for r in records if r.get("approved") is False]
        assert len(blocked) >= 1

    def test_extra_kwargs_stored(self):
        audit_order("BUY", "AAPL", 1, 100.0, "tech", "ok", True, needs_approval=True)
        records = read_audit_log(event_types=["ORDER"])
        r = records[-1]
        assert r.get("needs_approval") is True


# ---------------------------------------------------------------------------
# audit_safety_event
# ---------------------------------------------------------------------------

class TestAuditSafetyEvent:

    def test_safety_event_fields(self):
        audit_safety_event("CIRCUIT_BREAKER", "L1 triggered", drawdown=0.02)
        records = read_audit_log(event_types=["SAFETY"])
        assert len(records) >= 1
        r = records[-1]
        assert r["event"] == "SAFETY"
        assert r["level"] == "CIRCUIT_BREAKER"
        assert r["message"] == "L1 triggered"
        assert r["drawdown"] == 0.02

    def test_info_level_safety_event(self):
        audit_safety_event("INFO", "SafetyLayer initialized")
        records = read_audit_log(event_types=["SAFETY"])
        assert any(r["level"] == "INFO" for r in records)


# ---------------------------------------------------------------------------
# audit_signal
# ---------------------------------------------------------------------------

class TestAuditSignal:

    def test_signal_fields(self):
        audit_signal("meme_momentum", "GME", "BUY", 0.85, "reddit_spike")
        records = read_audit_log(event_types=["SIGNAL"])
        assert len(records) >= 1
        r = records[-1]
        assert r["event"] == "SIGNAL"
        assert r["strategy"] == "meme_momentum"
        assert r["symbol"] == "GME"
        assert r["signal"] == "BUY"
        assert r["score"] == 0.85
        assert r["source"] == "reddit_spike"


# ---------------------------------------------------------------------------
# audit_param_change
# ---------------------------------------------------------------------------

class TestAuditParamChange:

    def test_param_change_fields(self):
        audit_param_change("MemeMomentum", "spike_threshold", 2.0, 2.5, "optimizer")
        records = read_audit_log(event_types=["PARAM_CHANGE"])
        assert len(records) >= 1
        r = records[-1]
        assert r["event"] == "PARAM_CHANGE"
        assert r["component"] == "MemeMomentum"
        assert r["param"] == "spike_threshold"
        assert r["old"] == 2.0
        assert r["new"] == 2.5
        assert r["reason"] == "optimizer"


# ---------------------------------------------------------------------------
# read_audit_log — filtering and limits
# ---------------------------------------------------------------------------

class TestReadAuditLog:

    def test_returns_empty_list_for_missing_date(self):
        records = read_audit_log(date="1999-01-01")
        assert records == []

    def test_event_type_filter(self):
        audit_order("BUY", "AAPL", 1, 100.0, "tech", "ok", True)
        audit_safety_event("INFO", "test safety")
        audit_signal("tech", "AAPL", "BUY", 0.8, "src")

        orders = read_audit_log(event_types=["ORDER"])
        assert all(r["event"] == "ORDER" for r in orders)

        safety = read_audit_log(event_types=["SAFETY"])
        assert all(r["event"] == "SAFETY" for r in safety)

    def test_multi_event_type_filter(self):
        audit_order("BUY", "AAPL", 1, 100.0, "tech", "ok", True)
        audit_safety_event("INFO", "test")
        audit_signal("tech", "AAPL", "BUY", 0.8, "src")

        records = read_audit_log(event_types=["ORDER", "SAFETY"])
        event_types = {r["event"] for r in records}
        assert "SIGNAL" not in event_types
        assert "ORDER" in event_types or "SAFETY" in event_types

    def test_limit_returns_last_n_records(self):
        for i in range(10):
            audit_system("TEST", f"msg-{i}")
        records = read_audit_log(limit=3)
        assert len(records) == 3
        # Should be the last 3
        assert records[-1]["message"] == "msg-9"

    def test_limit_with_event_type_filter(self):
        for i in range(5):
            audit_order("BUY", "AAPL", i, 100.0, "tech", "ok", True)
        for i in range(5):
            audit_safety_event("INFO", f"safety-{i}")

        orders_limited = read_audit_log(event_types=["ORDER"], limit=2)
        assert len(orders_limited) == 2
        assert all(r["event"] == "ORDER" for r in orders_limited)

    def test_malformed_lines_are_skipped(self, temp_audit_dir):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = temp_audit_dir / f"audit_{today}.jsonl"
        with open(path, "a") as f:
            f.write("not valid json\n")
            f.write('{"event": "TEST", "message": "valid", "_ts": "x"}\n')
        records = read_audit_log()
        assert any(r.get("message") == "valid" for r in records)
