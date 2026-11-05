"""Checks diagnostic indicators without running or modifying any inference job."""
import unittest

from .diagnostic_compare import flags_for, matrix


VIEW = {"context": {"installed_apps": {}, "known_id": "evt_known"}, "history": [], "latest_text": "đổi giờ"}
TOOLS = {"calendar.update": {"parameters": {"properties": {}}}}


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def record(calls, observations=None, status="finished"):
    observations = observations or [{} for c in calls]
    return {"tool_calls": calls, "status": status, "trace": [
        {"executed": True, "proposal": c, "observation": o} for c, o in zip(calls, observations)]}


class DiagnosticTests(unittest.TestCase):
    def test_valid_repeated_gold_call_is_not_flagged_as_loop(self):
        calls = [call("alarm.list"), call("alarm.list")]
        result = flags_for(record(calls), {"tool_calls": calls}, VIEW, TOOLS)
        self.assertEqual(result["flags"], [])
        self.assertEqual(result["first_sequence_divergence"], "none")

    def test_id_from_previous_observation_or_context_is_visible(self):
        calls = [call("calendar.list"), call("calendar.update", event_id="evt_from_result"),
                 call("calendar.update", event_id="evt_known")]
        result = flags_for(record(calls, [{"events": [{"id": "evt_from_result"}]}, {}, {}]),
                           {"tool_calls": calls}, VIEW, TOOLS)
        self.assertNotIn("id_argument_not_seen_in_visible_input", result["flags"])
        unseen = flags_for(record([call("calendar.update", event_id="invented")]),
                           {"tool_calls": []}, VIEW, TOOLS)
        self.assertIn("id_argument_not_seen_in_visible_input", unseen["flags"])

    def test_arguments_early_finish_and_ask_indicators(self):
        pred = [call("calendar.update", event_id="evt_known", start="09:00")]
        gold = [call("calendar.update", event_id="evt_known", start="10:00"), call("sys.ask_missing_info")]
        result = flags_for(record(pred), {"tool_calls": gold}, VIEW, TOOLS)
        self.assertEqual(result["first_sequence_divergence"], "arguments")
        for flag in ["argument_not_matching_any_same_name_gold_call", "finished_with_fewer_calls_than_gold",
                     "missing_required_ask", "aligned_date_time_argument_mismatch"]:
            self.assertIn(flag, result["flags"])

    def test_extra_repetition_and_unsuccessful_tool_result(self):
        c = call("alarm.list")
        result = flags_for(record([c, c], [{"ok": True, "done": False}, {"ok": False}]),
                           {"tool_calls": [c]}, VIEW, TOOLS)
        for flag in ["repeated_call_beyond_gold_multiplicity", "tool_returned_ok_false", "tool_returned_done_false"]:
            self.assertIn(flag, result["flags"])

    def test_paired_outcomes_include_partial_scores_and_regressions(self):
        ids = list("abcde")
        g, a = [50, 100, 0, 0, 25], [25, 0, 100, 0, 25]
        def run(values):
            return {"scores": {i: {"score_spec_local": v} for i, v in zip(ids, values)}}
        result = matrix(run(g), run(a), ids)
        self.assertEqual(sum(result.get(k, 0) for k in ["both_positive", "gold_positive_asr_zero", "gold_zero_asr_positive", "both_zero"]), 5)
        self.assertEqual(result["gold_higher"], 2)
        self.assertEqual(result["asr_higher"], 1)
        self.assertEqual(result["equal"], 2)


if __name__ == "__main__":
    unittest.main()
