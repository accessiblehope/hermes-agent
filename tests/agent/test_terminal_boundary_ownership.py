import time
from unittest.mock import patch

import tools.process_registry as process_registry_module
from agent.terminal_boundary import enforce_public_terminal_response
from tools.process_registry import ProcessRegistry, ProcessSession


OWNER = "objective-owner-1"
SESSION = "hermes-session-1"


def _terminal_result(**extra):
    result = {
        "final_response": "DONE\nThe bounded objective has been handled.",
        "completed": True,
        "failed": False,
    }
    result.update(extra)
    return result


def _registry_with_running(monkeypatch, *, notify_on_complete=False):
    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_owned_test",
        command="sleep 60",
        task_id=OWNER,
        owner_task_id=OWNER,
        session_key=SESSION,
        started_at=time.time(),
        notify_on_complete=notify_on_complete,
    )
    registry._running[session.id] = session
    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    return registry, session


def test_terminal_transition_is_rejected_while_owned_background_work_runs(monkeypatch):
    registry, session = _registry_with_running(monkeypatch, notify_on_complete=True)

    result = enforce_public_terminal_response(
        _terminal_result(),
        platform="webui",
        task_id=OWNER,
    )

    receipt = result["core_control"]
    assert receipt["terminal"] is False
    assert receipt["owned_running_work"] == 1
    assert receipt["owned_work"][0]["session_id"] == session.id
    assert registry.running_owned_by(OWNER) == [session]


def test_requested_termination_revokes_owned_work_before_terminal_state(monkeypatch):
    registry, session = _registry_with_running(monkeypatch)

    def fake_kill(session_id, *, source, consume_output):
        assert session_id == session.id
        assert source == "objective.terminate"
        assert consume_output is True
        session.exited = True
        session.exit_code = -15
        registry._running.pop(session.id)
        registry._finished[session.id] = session
        return {"status": "killed"}

    monkeypatch.setattr(registry, "kill_process", fake_kill)

    result = enforce_public_terminal_response(
        _terminal_result(termination_requested=True),
        platform="webui",
        task_id=OWNER,
    )

    receipt = result["core_control"]
    assert receipt["terminal"] is True
    assert receipt["owned_running_work"] == 0
    assert receipt["owned_work"] == []
    assert registry.running_owned_by(OWNER) == []


def test_natural_completion_reconciles_running_ownership_without_duplicate_continuation(monkeypatch):
    registry, session = _registry_with_running(monkeypatch, notify_on_complete=True)
    session.exited = True
    session.exit_code = 0
    with patch.object(registry, "_write_checkpoint"):
        registry._move_to_finished(session)

    result = enforce_public_terminal_response(
        _terminal_result(),
        platform="webui",
        task_id=OWNER,
    )

    receipt = result["core_control"]
    assert receipt["terminal"] is True
    assert receipt["owned_running_work"] == 0
    assert receipt["owned_queued_work"] == 0
    assert registry.completion_queue.qsize() == 1
    assert registry.running_owned_by(OWNER) == []


def test_registry_discovery_denies_quiescence_when_session_bookkeeping_claims_none(monkeypatch):
    registry, _session = _registry_with_running(monkeypatch)

    result = enforce_public_terminal_response(
        _terminal_result(),
        platform="webui",
        task_id=OWNER,
    )

    receipt = result["core_control"]
    assert receipt["terminal"] is False
    assert receipt["owned_running_work"] == 1
