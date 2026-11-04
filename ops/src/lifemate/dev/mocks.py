"""Fictional tools and scripted backend; no official ToolEnv or state hash."""
from __future__ import annotations

from copy import deepcopy
from lifemate.runtime.contracts import (
    Calls, Execution, ModelRequest, Observation, ProposedCall, Validation,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    def respond(self, request: ModelRequest):
        self.requests.append(deepcopy(request))
        if not self.responses:
            raise RuntimeError("mock script exhausted")
        response = self.responses.pop(0)
        return response(request) if callable(response) else deepcopy(response)


class MockEnvironment:
    def __init__(self):
        self.calls: list[ProposedCall] = []
        self.updated = False

    def validate(self, call: ProposedCall) -> Validation:
        schemas = {
            "demo.lookup": {"query": str},
            "demo.update": {"record_id": str, "value": str},
            "demo.ask": {"question": str},
            "demo.pending": {},
        }
        schema = schemas.get(call.name)
        valid = schema is not None and set(call.arguments) == set(schema)
        if valid:
            valid = all(type(call.arguments[k]) is t for k, t in schema.items())
        return Validation(bool(valid), call.name == "demo.ask", "accepted" if valid else "schema_rejected")

    def execute(self, call: ProposedCall) -> Execution:
        self.calls.append(deepcopy(call))
        if call.name == "demo.lookup":
            obs = Observation(True, True, {"record_id": "mock-record-7"})
        elif call.name == "demo.update":
            self.updated = call.arguments["record_id"] == "mock-record-7"
            obs = Observation(True, self.updated, {"updated": self.updated})
        elif call.name == "demo.ask":
            obs = Observation(True, False, {"awaiting_user": True})
        else:
            obs = Observation(True, False, {"message": "mock work remains pending"})
        return Execution(True, obs)


def lookup_then_update(request: ModelRequest):
    # Uses the actual prior observation rather than an invented ID in the runner.
    record_id = request.observations[-1].execution.observation.data["record_id"]
    return Calls((ProposedCall("demo.update", {"record_id": record_id, "value": "ready"}),))


def scenarios():
    """Each tuple has its own clock so deterministic deadline demo needs no sleep."""
    from lifemate.runtime.contracts import ModelError
    clock = FakeClock()

    def late_response(request):
        clock.advance(request.remaining_seconds)
        return Calls((ProposedCall("demo.lookup", {"query": "late"}),))

    return [
        ("lookup_update", ScriptedModel([
            Calls((ProposedCall("demo.lookup", {"query": "example"}),)),
            lookup_then_update, Calls(())]), None),
        ("ask_stop", ScriptedModel([
            Calls((ProposedCall("demo.ask", {"question": "Which record?"}),)),
            Calls((ProposedCall("demo.update", {"record_id": "unused", "value": "blocked"}),)),
        ]), None),
        ("finish_no_action", ScriptedModel([Calls(())]), None),
        ("invalid_schema", ScriptedModel([Calls((ProposedCall("demo.update", {"value": "missing ID"}),))] * 3), None),
        ("model_error", ScriptedModel([ModelError()]), None),
        ("deadline", ScriptedModel([late_response]), clock),
        ("parse_repair", ScriptedModel([ModelError("parse_error"), Calls(())]), None),
        ("not_done", ScriptedModel([Calls((ProposedCall("demo.pending", {}),)), Calls(())]), None),
    ]
