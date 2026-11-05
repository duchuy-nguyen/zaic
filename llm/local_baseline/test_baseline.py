"""Meaningful regressions for score matching and one-session execution boundaries."""
from __future__ import annotations

import json
import time
import unittest

from .agent import infer, parse_decision
from .data import agent_view
from .scoring import compatible, maximum_matches, score, token_f1
from .speech import audio_view, normalize_transcript


TOOLS = {"sms.send": {"parameters": {"properties": {"content": {"type": "string", "x-match": "fuzzy"}}}}}


def call(text):
    return {"name": "sms.send", "arguments": {"content": text}}


def response(name=None, arguments=None):
    message = {"role": "assistant", "content": ""}
    if name:
        message["tool_calls"] = [{"id": "test_call", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments or {})}}]
    return {"choices": [{"finish_reason": "tool_calls" if name else "stop", "message": message}]}


class ScorerTests(unittest.TestCase):
    def test_bipartite_reassignment_not_greedy(self):
        self.assertEqual(maximum_matches([call("a b"), call("a c")], [call("a b c"), call("a b d")], TOOLS), 2)

    def test_duplicates_cannot_reuse_a_gold_call(self):
        self.assertEqual(maximum_matches([call("hello"), call("hello")], [call("hello")], TOOLS), 1)

    def test_threshold_is_strictly_greater(self):
        self.assertAlmostEqual(token_f1("a b c d e", "a b c x y"), .6)
        self.assertFalse(compatible(call("a b c d e"), call("a b c x y"), TOOLS))
        self.assertTrue(compatible(call("a b c d e"), call("a b c d y"), TOOLS))

    def test_fuzzy_only_for_marked_fields_and_exact_key_sets(self):
        self.assertFalse(compatible(call("a b"), call("a b c"), {}))
        extra = call("hello")
        extra["arguments"]["contact_id"] = "invented"
        self.assertFalse(compatible(extra, call("hello"), TOOLS))

    def test_state_gate_and_timeout_override(self):
        gold = {"tool_calls": [call("hello")], "state_hash": "correct"}
        bad = {"tool_calls": [call("hello")], "state_hash": "wrong"}
        self.assertEqual(score(bad, gold, TOOLS)["score_spec_local"], 0)
        empty = {"tool_calls": [], "state_hash": "same"}
        self.assertEqual(score(empty, empty, TOOLS, failed=True)["score_spec_local"], 0)


class FakeEnv:
    def __init__(self, valid=True):
        self.calls = []
        self.valid = valid

    def validate(self, name, arguments):
        return {"ok": self.valid, "error": "invalid"}

    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": arguments})
        return {"ok": True, "done": True}

    def get_tool_calls(self):
        return list(self.calls)

    def get_state_hash(self):
        return "test_hash"


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = 0

    def generate(self, messages, tools, deadline):
        self.requests += 1
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output, 100


class AgentTests(unittest.TestCase):
    def view(self):
        return {"id": "sample", "latest_text": "hi", "context": {"installed_apps": {}}, "history": []}

    def test_labels_never_enter_agent_view(self):
        row = {"id": "x", "audio_text": "hello", "context": {}, "history": [],
               "label": {"secret": "GOLD_SECRET"}, "final_state": {"secret": "STATE_SECRET"}}
        self.assertNotIn("SECRET", json.dumps(agent_view(row)))

    def test_audio_view_uses_asr_for_current_and_history_without_gold(self):
        row = {"id": "x", "audio": "audio/current.wav", "audio_text": "GOLD_CURRENT_SECRET",
               "context": {}, "label": {"secret": "LABEL_SECRET"},
               "history": [{"role": "user", "audio": "audio/history.wav", "audio_text": "GOLD_HISTORY_SECRET"}]}
        view = audio_view(row, {"audio/current.wav": "ASR current", "audio/history.wav": "ASR history"})
        self.assertEqual(view["latest_text"], "ASR current")
        self.assertEqual(view["history"][0]["text"], "ASR history")
        self.assertNotIn("SECRET", json.dumps(view))

    def test_audio_view_missing_history_never_falls_back_to_gold(self):
        row = {"id": "x", "audio": "current.wav", "context": {}, "audio_text": "gold",
               "history": [{"role": "user", "audio": "history.wav", "audio_text": "gold history"}]}
        with self.assertRaises(ValueError):
            audio_view(row, {"current.wav": "ASR current"})

    def test_normalization_does_not_rewrite_entities(self):
        self.assertEqual(normalize_transcript("  An   chuyển 050 nghìn  "), "An chuyển 050 nghìn")

    def test_ask_executes_once_then_stops(self):
        backend = FakeBackend([response("sys.ask_missing_info", {"tool_name": "sms.send", "params": ["contact_id"]})])
        env = FakeEnv()
        result = infer(self.view(), {"sys.ask_missing_info": {"parameters": {}}}, env, backend)
        self.assertEqual(result["status"], "asked_missing_info")
        self.assertEqual(len(env.calls), 1)
        self.assertEqual(backend.requests, 1)

    def test_schema_repairs_never_execute_invalid_proposals(self):
        backend = FakeBackend([response("sys.ask_missing_info")] * 3)
        env = FakeEnv(valid=False)
        result = infer(self.view(), {"sys.ask_missing_info": {"parameters": {}}}, env, backend)
        self.assertEqual(result["status"], "structured_output_error")
        self.assertEqual(env.calls, [])

    def test_timeout_is_not_silent_finish(self):
        backend = FakeBackend([TimeoutError("deadline")])
        result = infer(self.view(), {"sys.ask_missing_info": {"parameters": {}}}, FakeEnv(), backend)
        self.assertEqual(result["status"], "timeout")

    def test_truncated_output_is_not_finish(self):
        raw = response()
        raw["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            parse_decision(raw)

    def test_missing_finish_reason_is_not_silent_success(self):
        raw = response()
        del raw["choices"][0]["finish_reason"]
        with self.assertRaises(ValueError):
            parse_decision(raw)

    def test_missing_schema_reports_failure_instead_of_aborting_job(self):
        result = infer(self.view(), {}, FakeEnv(), FakeBackend([]))
        self.assertEqual(result["status"], "execution_error")

    def test_multiple_proposals_are_not_silently_dropped(self):
        raw = response("sys.ask_missing_info")
        raw["choices"][0]["message"]["tool_calls"] *= 2
        with self.assertRaises(ValueError):
            parse_decision(raw)


if __name__ == "__main__":
    unittest.main()
