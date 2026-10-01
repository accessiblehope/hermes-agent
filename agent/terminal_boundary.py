"""External Core Control application adapter for Hermes turn metadata."""
from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

_INTERNAL_PLATFORMS = frozenset({"subagent", "internal", "tool"})
_CORE_CONTROL_ROOT = Path("/Users/peterbeddow/Core/core-control")


def _terminal_state_requested(result: dict[str, Any]) -> bool:
    """Whether this turn is attempting a governed terminal transition."""
    if result.get("completed") is True:
        return True
    if result.get("termination_requested") is True:
        return True
    return str(result.get("lifecycle_state") or "").upper() in {
        "BLOCKED", "DONE", "QUIESCENT", "TERMINATED"
    }


def _execution_snapshot(*, task_id: str, termination_requested: bool) -> dict[str, Any]:
    """Reconcile the existing process registry before terminal disposition."""
    from tools.process_registry import process_registry

    snapshot = process_registry.reconcile_owned_execution(task_id)
    if termination_requested and (
        snapshot.get("owned_running_work", 0) or snapshot.get("owned_queued_work", 0)
    ):
        snapshot = process_registry.terminate_owned_execution(task_id)
    return snapshot


def enforce_public_terminal_response(
    result: dict[str, Any], *, platform: str = "", task_id: str | None = None,
    termination_requested: bool | None = None,
) -> dict[str, Any]:
    """Ask Core Control for a disposition after reconciling owned execution.

    The Hermes process registry is authoritative for live background work. Core
    Control receives that reconciled snapshot and rejects terminal/quiescent
    disposition while owned execution remains live.
    """
    if not isinstance(result, dict) or platform in _INTERNAL_PLATFORMS:
        return result

    requested_termination = bool(
        result.get("termination_requested")
        if termination_requested is None
        else termination_requested
    )
    owner = str(task_id or result.get("task_id") or "")
    terminal_requested = _terminal_state_requested(result)
    ownership = None
    ownership_blocked = False
    if owner and terminal_requested:
        ownership = _execution_snapshot(
            task_id=owner, termination_requested=requested_termination,
        )
        ownership_blocked = bool(
            ownership.get("owned_running_work", 0)
            or ownership.get("owned_queued_work", 0)
        )
        ownership["termination_requested"] = requested_termination

    request = {
        "protocol_version": "hermes-core-control/v1",
        "request_id": uuid.uuid4().hex,
        "source": "hermes",
        "platform": platform,
        "task_id": owner,
        "termination_requested": requested_termination,
        "execution_ownership": ownership,
        "turn_result": result,
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "core_control", "validate-public-response", json.dumps(request)],
            cwd=_CORE_CONTROL_ROOT,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Core Control exited with status {completed.returncode}")
        disposition = json.loads(completed.stdout)
        if not isinstance(disposition, dict):
            raise ValueError("Core Control returned a non-object disposition")
        result["core_control"] = {
            "protocol_version": disposition.get("protocol_version"),
            "request_id": disposition.get("request_id"),
            "outcome": disposition.get("disposition"),
            "terminal": disposition.get("terminal") is True,
            "verified": disposition.get("verified") is True,
            "response_preserved": disposition.get("response_preserved") is True,
            "continuation_required": disposition.get("continuation_required") is True,
            "owned_running_work": disposition.get("owned_running_work", 0),
            "owned_queued_work": disposition.get("owned_queued_work", 0),
            "owned_work": disposition.get("owned_work", []),
            "termination_confirmed": disposition.get("termination_confirmed"),
        }
    except Exception as exc:
        # Core Control failure must not erase a provider response, but it must
        # fail closed when the local authoritative registry found live work.
        result["core_control"] = {
            "protocol_version": request["protocol_version"],
            "request_id": request["request_id"],
            "terminal": False,
            "available": False,
            "error": type(exc).__name__,
            "continuation_required": ownership_blocked,
            "owned_running_work": ownership.get("owned_running_work", 0) if ownership else 0,
            "owned_queued_work": ownership.get("owned_queued_work", 0) if ownership else 0,
            "owned_work": ownership.get("owned_work", []) if ownership else [],
        }
    return result
