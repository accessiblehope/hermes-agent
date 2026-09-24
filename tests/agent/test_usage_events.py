"""Invariant tests for durable per-response usage events."""

from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION


class _Compressor:
    threshold_tokens = 100_000
    context_length = 200_000
    awaiting_real_usage_after_compression = False
    _verify_compaction_cleared_threshold = False

    def update_from_response(self, usage):
        self.last_usage = usage


def _agent(db):
    return SimpleNamespace(
        context_compressor=_Compressor(),
        provider="provider-test",
        api_mode="chat_completions",
        model="model-test",
        base_url="https://provider.example/v1",
        api_key="",
        session_id="session-test",
        client=None,
        _session_db=db,
        _session_db_created=True,
        _persist_disabled=False,
        _current_task_id="task-test",
        _current_api_request_id="turn-test:api:1",
        session_api_calls=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_estimated_cost_usd=0.0,
        session_cost_status=None,
        session_cost_source=None,
        verbose_logging=False,
        quiet_mode=True,
        log_prefix="",
        _api_latency_history=[],
        _api_output_history=[],
        _safe_print=lambda *args, **kwargs: None,
        _vprint=lambda *args, **kwargs: None,
    )


def _response(response_id="response-test"):
    return SimpleNamespace(
        id=response_id,
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            cache_read_input_tokens=3,
            cache_creation_input_tokens=2,
            output_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ),
    )


def _record(agent, response):
    from agent import turn_usage

    return turn_usage.record_response_usage(
        agent,
        response,
        messages=[{"role": "user", "content": "hi"}],
        api_call_count=1,
        api_duration=0.2,
        compression_attempts=0,
        max_compression_attempts=3,
    )


@pytest.fixture()
def db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    database.create_session("session-test", "cli", model="model-test")
    yield database
    database.close()


def test_fresh_schema_has_usage_events_and_current_version(db):
    columns = {
        row[1] for row in db._conn.execute("PRAGMA table_info(usage_events)").fetchall()
    }

    assert {
        "event_id", "session_id", "occurred_at", "provider_response_id", "model",
        "billing_provider", "billing_base_url", "billing_mode", "task",
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
        "reasoning_tokens", "estimated_cost_usd", "actual_cost_usd", "cost_status",
        "cost_source", "pricing_version",
    } <= columns
    indexes = {
        row[1] for row in db._conn.execute(
            "SELECT type, name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'usage_events'"
        ).fetchall()
    }
    assert {
        "idx_usage_events_session_time", "idx_usage_events_occurred_at",
        "idx_usage_events_provider_response_id",
    } <= indexes
    assert db._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION


def test_existing_schema_migrates_usage_events_without_losing_rows(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.create_session("legacy-session", "cli", model="legacy-model")
    db.update_token_counts("legacy-session", model="legacy-model", input_tokens=4, api_call_count=1)
    db._execute_write(
        lambda conn: (
            conn.execute("DROP TABLE usage_events"),
            conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,)),
        )
    )
    db.close()

    migrated = SessionDB(db_path=path)
    try:
        assert migrated._conn.execute("SELECT 1 FROM usage_events LIMIT 1").fetchone() is None
        assert migrated.get_session("legacy-session")["input_tokens"] == 4
        assert migrated._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        migrated.close()


def test_response_usage_persists_exact_event_fields_and_timestamp(db, monkeypatch):
    from agent import turn_usage

    monkeypatch.setattr(turn_usage.time, "time", lambda: 1_700_000_000.25)
    monkeypatch.setattr(
        turn_usage,
        "estimate_usage_cost",
        lambda *args, **kwargs: SimpleNamespace(
            amount_usd=Decimal("0.1234"), status="estimated", source="unit-test",
            pricing_version="pricing-test-v1",
        ),
    )

    _record(_agent(db), _response())
    assert db.flush_token_counts()
    event = dict(db._conn.execute("SELECT * FROM usage_events").fetchone())

    assert event == {
        "event_id": "response-test",
        "session_id": "session-test",
        "occurred_at": 1_700_000_000.25,
        "provider_response_id": "response-test",
        "model": "model-test",
        "billing_provider": "provider-test",
        "billing_base_url": "https://provider.example/v1",
        "billing_mode": "chat_completions",
        "task": "task-test",
        "input_tokens": 6,
        "output_tokens": 7,
        "cache_read_tokens": 3,
        "cache_write_tokens": 2,
        "reasoning_tokens": 5,
        "estimated_cost_usd": 0.1234,
        "actual_cost_usd": None,
        "cost_status": "estimated",
        "cost_source": "unit-test",
        "pricing_version": "pricing-test-v1",
    }
    session = db.get_session("session-test")
    assert session["input_tokens"] == 6
    assert session["output_tokens"] == 7
    usage = db._conn.execute(
        "SELECT input_tokens, output_tokens, api_call_count FROM session_model_usage"
    ).fetchone()
    assert tuple(usage) == (6, 7, 1)


def test_repeated_response_persistence_is_idempotent(db, monkeypatch):
    from agent import turn_usage

    monkeypatch.setattr(
        turn_usage,
        "estimate_usage_cost",
        lambda *args, **kwargs: SimpleNamespace(
            amount_usd=Decimal("0.01"), status="estimated", source="unit-test",
            pricing_version="pricing-test-v1",
        ),
    )
    agent = _agent(db)
    response = _response()

    _record(agent, response)
    _record(agent, response)

    assert db._conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0] == 1


def test_response_without_id_uses_a_stable_uuid_event_id(db, monkeypatch):
    from agent import turn_usage

    monkeypatch.setattr(
        turn_usage,
        "estimate_usage_cost",
        lambda *args, **kwargs: SimpleNamespace(
            amount_usd=Decimal("0.01"), status="estimated", source="unit-test",
            pricing_version="pricing-test-v1",
        ),
    )
    agent = _agent(db)
    response = _response(response_id=None)

    _record(agent, response)
    _record(agent, response)
    rows = db._conn.execute(
        "SELECT event_id, provider_response_id FROM usage_events"
    ).fetchall()

    assert len(rows) == 1
    UUID(rows[0][0])
    assert rows[0][1] is None


def test_no_usage_response_does_not_create_usage_event(db):
    _record(_agent(db), SimpleNamespace(id="no-usage", usage=None))

    assert db._conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0] == 0
