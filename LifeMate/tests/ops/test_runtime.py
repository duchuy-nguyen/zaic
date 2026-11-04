from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from lifemate.dev.demo import main
from lifemate.dev.mocks import FakeClock, MockEnvironment, ScriptedModel, lookup_then_update, scenarios
from lifemate.runtime.contracts import (
    Calls, Execution, ModelError, Observation, ProposedCall,
    RuntimeConfig, Sample, StopReason, Validation,
)
from lifemate.runtime.runner import Runner
from lifemate.runtime.tracing import JsonlWriter


def call(name="demo.lookup", **args):
    return ProposedCall(name, args or {"query": "example"})


def batch(*calls):
    return Calls(tuple(calls))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.environments = []
        self.events = []

    def factory(self, sample):
        env = MockEnvironment()
        self.environments.append(env)
        return env

    def run_model(self, responses, **kwargs):
        self.model = ScriptedModel(responses)
        return Runner(self.factory, trace=self.events.append, **kwargs).run(Sample("sample"), self.model)

    def test_observation_dependency_and_order(self):
        result = self.run_model([batch(call()), lookup_then_update, Calls(())])
        self.assertEqual(result.stop_reason, StopReason.FINISH)
        self.assertEqual([r.proposal.name for r in result.executed_calls], ["demo.lookup", "demo.update"])
        self.assertEqual(self.environments[0].calls[1].arguments["record_id"], "mock-record-7")
        self.assertTrue(self.environments[0].updated)
        self.assertEqual(self.model.requests[1].observations[0].execution.observation.data,
                         {"record_id": "mock-record-7"})

    def test_samples_have_separate_environment_and_no_history_replay(self):
        runner = Runner(self.factory)
        old_call = {"name": "demo.update", "arguments": {"record_id": "old", "value": "old"}}
        runner.run(Sample("a", {"history": [{"role": "assistant", "tool_calls": [old_call]}]}),
                   ScriptedModel([batch(call()), lookup_then_update, Calls(())]))
        runner.run(Sample("b"), ScriptedModel([Calls(())]))
        self.assertIsNot(self.environments[0], self.environments[1])
        self.assertEqual(len(self.environments[0].calls), 2)
        self.assertEqual(self.environments[1].calls, [])
        self.assertFalse(self.environments[1].updated)

    def test_ask_blocks_later_model_turns(self):
        result = self.run_model([batch(call()), batch(call("demo.ask", question="which?")), batch(call())])
        self.assertEqual(result.stop_reason, StopReason.ASK)
        self.assertEqual([c.name for c in self.environments[0].calls], ["demo.lookup", "demo.ask"])
        self.assertEqual(len(self.model.requests), 2)
        self.assertEqual(sum(e["event"] == "execution" for e in self.events), 2)

    def test_schema_rejection_never_executes(self):
        result = self.run_model([batch(call("demo.update", value="missing ID"))],
                                config=RuntimeConfig(max_repairs=0))
        self.assertEqual(result.stop_reason, StopReason.REPAIR_BUDGET)
        self.assertEqual(self.environments[0].calls, [])
        self.assertEqual(result.executed_calls, ())
        self.assertTrue(result.failure)

    def test_bad_shape_never_reaches_environment(self):
        result = self.run_model([batch(ProposedCall("demo.lookup", "not a dict"))],
                                config=RuntimeConfig(max_repairs=0))
        self.assertEqual(result.stop_reason, StopReason.REPAIR_BUDGET)
        self.assertEqual(self.environments[0].calls, [])

    def test_backend_deadline_has_no_late_execution(self):
        clock = FakeClock()
        def late(request):
            self.assertEqual(request.remaining_seconds, 1)
            clock.advance(1)
            return batch(call())
        result = self.run_model([late], config=RuntimeConfig(sample_deadline_seconds=1), clock=clock)
        self.assertEqual(result.stop_reason, StopReason.DEADLINE)
        self.assertEqual(self.environments[0].calls, [])

    def test_factory_time_counts_toward_whole_sample(self):
        clock = FakeClock()
        def factory(sample):
            clock.advance(2)
            return self.factory(sample)
        model = ScriptedModel([Calls(())])
        result = Runner(factory, RuntimeConfig(sample_deadline_seconds=1), clock=clock).run(Sample("s"), model)
        self.assertEqual(result.stop_reason, StopReason.DEADLINE)
        self.assertEqual(model.requests, [])

    def test_deadline_during_validation_blocks_execution(self):
        clock = FakeClock()
        class SlowValidation(MockEnvironment):
            def validate(self, proposed):
                clock.advance(1)
                return super().validate(proposed)
        env = SlowValidation()
        result = Runner(lambda sample: env, RuntimeConfig(sample_deadline_seconds=1), clock=clock).run(
            Sample("s"), ScriptedModel([batch(call())]))
        self.assertEqual(result.stop_reason, StopReason.DEADLINE)
        self.assertEqual(env.calls, [])

    def test_deadline_after_execution_preserves_first(self):
        clock = FakeClock()
        class SlowExecution(MockEnvironment):
            def execute(self, proposed):
                value = super().execute(proposed)
                clock.advance(1)
                return value
        env = SlowExecution()
        result = Runner(lambda sample: env, RuntimeConfig(sample_deadline_seconds=1), clock=clock).run(
            Sample("s"), ScriptedModel([batch(call()), batch(call())]))
        self.assertEqual(result.stop_reason, StopReason.DEADLINE)
        self.assertEqual(len(result.executed_calls), 1)
        self.assertEqual(len(env.calls), 1)

    def test_deadline_before_execute_includes_trace_time(self):
        clock = FakeClock()
        def trace(event):
            if event["event"] == "validation":
                clock.advance(1)
        result = Runner(self.factory, RuntimeConfig(sample_deadline_seconds=1), clock=clock, trace=trace).run(
            Sample("s"), ScriptedModel([batch(call())]))
        self.assertEqual(result.stop_reason, StopReason.DEADLINE)
        self.assertEqual(self.environments[0].calls, [])

    def test_prior_executed_calls_survive_model_and_parse_errors(self):
        for error in (ModelError(), ModelError("parse_error")):
            result = self.run_model([batch(call()), error], config=RuntimeConfig(max_repairs=0))
            self.assertEqual(result.stop_reason, StopReason.REPAIR_BUDGET if error.code == "parse_error" else StopReason.MODEL_ERROR)
            self.assertEqual(len(result.executed_calls), 1)
            self.assertTrue(result.failure)

    def test_unknown_execution_not_retried(self):
        class Broken(MockEnvironment):
            def execute(self, proposed):
                value = super().execute(proposed)
                if len(self.calls) == 2:
                    raise RuntimeError("api_key=do-not-log")
                return value
        env = Broken()
        result = Runner(lambda sample: env, trace=self.events.append).run(
            Sample("s"), ScriptedModel([batch(call()), batch(call()), batch(call())]))
        self.assertEqual(result.stop_reason, StopReason.EXECUTION_UNKNOWN)
        self.assertEqual(len(result.executed_calls), 1)
        self.assertEqual(len(result.uncertain_calls), 1)
        self.assertEqual(len(env.calls), 2)
        self.assertNotIn("do-not-log", str(self.events))

    def test_explicit_unknown_status_stops_next_turn(self):
        class Unknown(MockEnvironment):
            def execute(self, proposed):
                self.calls.append(proposed)
                return Execution(None, Observation(False, False))
        env = Unknown()
        result = Runner(lambda sample: env).run(Sample("s"), ScriptedModel([batch(call()), batch(call())]))
        self.assertEqual(result.stop_reason, StopReason.EXECUTION_UNKNOWN)
        self.assertEqual(len(env.calls), 1)
        self.assertEqual(len(result.uncertain_calls), 1)

    def test_tool_error_is_preserved_and_model_decides_without_retry(self):
        class ToolError(MockEnvironment):
            def execute(self, proposed):
                self.calls.append(proposed)
                return Execution(True, Observation(False, False, {"error": "write rejected"}, terminal=True))
        env = ToolError()
        def inspect(request):
            self.assertEqual(request.observations[-1].execution.observation,
                             Observation(False, False, {"error": "write rejected"}, terminal=True))
            self.assertIsNone(request.repair)
            return Calls(())
        result = Runner(lambda sample: env, trace=self.events.append).run(
            Sample("s"), ScriptedModel([batch(call("demo.update", record_id="id", value="v")), inspect]))
        self.assertEqual(result.stop_reason, StopReason.FINISH)
        self.assertEqual(len(env.calls), 1)
        self.assertEqual(len(result.executed_calls), 1)
        self.assertFalse(next(e for e in self.events if e["event"] == "execution")["observation"].ok)

    def test_done_false_reaches_model_without_success_inference(self):
        def inspect(request):
            obs = request.observations[-1].execution.observation
            self.assertTrue(obs.ok)
            self.assertFalse(obs.done)
            return Calls(())
        result = self.run_model([batch(ProposedCall("demo.pending", {})), inspect])
        self.assertFalse(result.last_observation.done)
        self.assertEqual(result.stop_reason, StopReason.FINISH)
        self.assertFalse(hasattr(result, "task_success"))

    def test_finish_has_no_actions(self):
        result = self.run_model([Calls(())])
        self.assertEqual(result.executed_calls, ())
        self.assertEqual(result.stop_reason, StopReason.FINISH)

    def test_call_budget_blocks_next_turn(self):
        result = self.run_model([batch(call()), batch(call())], config=RuntimeConfig(max_executed_calls=1))
        self.assertEqual(result.stop_reason, StopReason.CALL_BUDGET)
        self.assertEqual(len(result.executed_calls), 1)

    def test_nonexecuted_observation_does_not_consume_call_budget(self):
        class Rejected(MockEnvironment):
            def execute(self, proposed):
                self.calls.append(proposed)
                return Execution(False, Observation(False, False))
        env = Rejected()
        model = ScriptedModel([batch(call()), Calls(())])
        result = Runner(lambda sample: env, RuntimeConfig(max_executed_calls=1)).run(Sample("s"), model)
        self.assertEqual(result.stop_reason, StopReason.FINISH)
        self.assertEqual(result.executed_calls, ())
        self.assertFalse(model.requests[1].observations[0].execution.executed)

    def test_repeated_nonexecution_is_bounded_by_steps(self):
        class Rejected(MockEnvironment):
            def execute(self, proposed):
                return Execution(False, Observation(False, False))
        model = ScriptedModel([batch(call())] * 3)
        result = Runner(lambda sample: Rejected(), RuntimeConfig(max_steps=2)).run(Sample("s"), model)
        self.assertEqual(result.stop_reason, StopReason.STEP_BUDGET)
        self.assertEqual(len(model.requests), 2)

    def test_empty_or_invalid_model_response_is_error_not_finish(self):
        for response in (Calls(42), None, {"finish": True}, []):
            result = self.run_model([response], config=RuntimeConfig(max_repairs=0))
            self.assertEqual(result.stop_reason, StopReason.REPAIR_BUDGET)
            self.assertTrue(result.failure)

    def test_backend_exception_is_failure_and_does_not_log_secrets(self):
        def broken(request):
            raise ValueError("password=private-value")
        result = self.run_model([batch(call()), broken])
        self.assertEqual(result.stop_reason, StopReason.MODEL_ERROR)
        self.assertEqual(len(result.executed_calls), 1)
        self.assertNotIn("private-value", str(self.events))

    def test_mutating_backend_cannot_erase_executed_calls(self):
        def mutate(request):
            request.observations[0].proposal.arguments.clear()
            request.observations[0].execution.observation.data.clear()
            return Calls(())
        result = self.run_model([batch(call()), mutate])
        self.assertEqual(result.executed_calls[0].proposal.arguments, {"query": "example"})
        self.assertEqual(result.last_observation.data, {"record_id": "mock-record-7"})

    def test_remaining_time_decreases_across_turns(self):
        clock = FakeClock()
        def first(request):
            clock.advance(2)
            return batch(call())
        self.run_model([first, Calls(())], config=RuntimeConfig(sample_deadline_seconds=5), clock=clock)
        self.assertEqual([r.remaining_seconds for r in self.model.requests], [5, 3])

    def test_factory_failure_produces_explicit_result(self):
        def broken(sample):
            raise RuntimeError("secret")
        result = Runner(broken).run(Sample("broken"), ScriptedModel([]))
        self.assertEqual(result.sample_id, "broken")
        self.assertEqual(result.stop_reason, StopReason.ENVIRONMENT_ERROR)
        self.assertTrue(result.failure)

    def test_parse_repair_feedback_precedes_any_environment_call(self):
        def repair(request):
            self.assertEqual(request.repair.kind, "parse_error")
            self.assertEqual(request.repairs_remaining, 1)
            self.assertEqual(request.observations, ())
            self.assertEqual(self.environments[0].calls, [])
            return batch(call())
        result = self.run_model([ModelError("parse_error"), repair, Calls(())])
        self.assertEqual(result.stop_reason, StopReason.FINISH)
        self.assertEqual(result.repairs_used, 1)
        self.assertIsNone(self.model.requests[-1].repair)
        self.assertEqual(len(result.executed_calls), 1)
        self.assertFalse(any(e["event"] == "validation" for e in self.events if e["step"] == 1))

    def test_validation_feedback_leaves_arguments_untouched(self):
        invalid = call("demo.update", value="missing ID")
        def repair(request):
            self.assertEqual(request.repair.kind, "validation_error")
            self.assertEqual(request.repair.proposal, invalid)
            self.assertFalse(request.repair.validation.valid)
            self.assertEqual(request.observations, ())
            self.assertEqual(self.environments[0].calls, [])
            return batch(call("demo.update", record_id="mock-record-7", value="fixed by model"))
        result = self.run_model([batch(invalid), repair, Calls(())])
        self.assertEqual(result.stop_reason, StopReason.FINISH)
        self.assertEqual(invalid.arguments, {"value": "missing ID"})
        self.assertEqual(len(result.executed_calls), 1)
        self.assertEqual(result.executed_calls[0].proposal.arguments["value"], "fixed by model")

    def test_parse_and_schema_repairs_share_whole_sample_budget(self):
        result = self.run_model([
            ModelError("parse_error"), batch(call("demo.update", value="bad")),
            batch(call()), ModelError("parse_error"), Calls(())])
        self.assertEqual(result.stop_reason, StopReason.REPAIR_BUDGET)
        self.assertEqual(result.repairs_used, 2)
        self.assertEqual(len(result.executed_calls), 1)
        self.assertEqual(len(self.model.requests), 4)
        self.assertEqual([r.repairs_remaining for r in self.model.requests], [2, 1, 0, 0])
        self.assertTrue(result.failure)

    def test_multi_call_response_executes_nothing_and_requests_repair(self):
        def repair(request):
            self.assertEqual(request.repair.kind, "protocol_error")
            self.assertEqual(self.environments[0].calls, [])
            return Calls(())
        result = self.run_model([batch(call(), call()), repair])
        self.assertEqual(result.executed_calls, ())
        self.assertEqual(result.repairs_used, 1)
        self.assertEqual(result.stop_reason, StopReason.FINISH)

    def test_parsed_empty_calls_finish_but_parse_errors_exhaust_budget(self):
        good = self.run_model([Calls(())])
        self.assertEqual(good.stop_reason, StopReason.FINISH)
        self.assertFalse(good.failure)
        bad = self.run_model([ModelError("parse_error")] * 3 + [Calls(())])
        self.assertEqual(bad.stop_reason, StopReason.REPAIR_BUDGET)
        self.assertTrue(bad.failure)
        self.assertEqual(len(self.model.requests), 3)

    def test_repaired_call_cannot_execute_after_deadline(self):
        clock = FakeClock()
        def late_repair(request):
            self.assertEqual(request.repair.kind, "parse_error")
            clock.advance(request.remaining_seconds)
            return batch(call())
        result = self.run_model([ModelError("parse_error"), late_repair], clock=clock)
        self.assertEqual(result.stop_reason, StopReason.DEADLINE)
        self.assertEqual(result.executed_calls, ())
        self.assertEqual(result.repairs_used, 1)

    def test_timeout_during_repair_is_never_empty_finish(self):
        def timeout(request):
            raise TimeoutError("transport timeout")
        result = self.run_model([ModelError("parse_error"), timeout, Calls(())])
        self.assertEqual(result.stop_reason, StopReason.MODEL_ERROR)
        self.assertTrue(result.failure)
        self.assertEqual(len(self.model.requests), 2)

    def test_official_ask_name_stops_after_execution(self):
        class AskEnv(MockEnvironment):
            def validate(self, proposed):
                return Validation(True)
        env = AskEnv()
        model = ScriptedModel([batch(call("sys.ask_missing_info", question="which?")), batch(call())])
        result = Runner(lambda sample: env).run(Sample("s"), model)
        self.assertEqual(result.stop_reason, StopReason.ASK)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(len(result.executed_calls), 1)

    def test_ask_not_executed_does_not_stop_sample(self):
        class RejectedAsk(MockEnvironment):
            def execute(self, proposed):
                return Execution(False, Observation(False, False))
        env = RejectedAsk()
        model = ScriptedModel([batch(call("demo.ask", question="which?")), Calls(())])
        result = Runner(lambda sample: env).run(Sample("s"), model)
        self.assertEqual(result.stop_reason, StopReason.FINISH)
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(result.executed_calls, ())

    def test_invalid_config_rejected(self):
        for kwargs in ({"max_repairs": -1}, {"max_repairs": True}, {"max_steps": 0}, {"max_executed_calls": True},
                       {"sample_deadline_seconds": float("nan")},
                       {"sample_deadline_seconds": -1}):
            with self.assertRaises(ValueError):
                RuntimeConfig(**kwargs)


class ArtifactTests(unittest.TestCase):
    def test_jsonl_redacts_nested_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            with JsonlWriter(path) as writer:
                writer({"proposal": ProposedCall("demo.lookup", {
                    "api_key": "private-key", "nested": {"password": "private-password"},
                    "value": "Bearer private-bearer"})})
            raw = path.read_text()
            self.assertNotIn("private-", raw)
            self.assertIn("[REDACTED]", raw)
            json.loads(raw)

    def test_offline_demo_has_all_results_trace_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            self.assertEqual(main(["--output-dir", str(output)]), 0)
            results = [json.loads(line) for line in (output / "mock_results.jsonl").read_text().splitlines()]
            expected = dict(lookup_update="finish", ask_stop="ask", finish_no_action="finish",
                            invalid_schema="repair_budget", model_error="model_error",
                            deadline="deadline", parse_repair="finish", not_done="finish")
            self.assertEqual({r["sample_id"]: r["stop_reason"] for r in results}, expected)
            self.assertEqual(len(results), 8)
            self.assertFalse(results[-1]["last_observation"]["done"])
            self.assertFalse((output / "submission.jsonl").exists())
            self.assertNotIn("state_hash", (output / "mock_results.jsonl").read_text())
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertFalse(manifest["official_submission"])
            self.assertEqual(manifest["config"]["output_tokens_per_step"], 512)
            self.assertEqual(len(manifest["code"]["source_sha256"]), 64)
            traces = [json.loads(line) for line in (output / "tool_traces.jsonl").read_text().splitlines()]
            stops = [e for e in traces if e["event"] == "stop"]
            self.assertEqual(len(stops), 8)
            for event in traces:
                self.assertTrue({"sample_id", "step", "proposal", "validation", "executed",
                                 "observation", "latency_seconds", "stop_reason"} <= set(event))
            with self.assertRaises(FileExistsError):
                main(["--output-dir", str(output)])


if __name__ == "__main__":
    unittest.main()
