"""Post-hoc paired clean-text/ASR diagnosis, never imported by live inference."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import statistics
import sys

from .agent import initial_messages
from .analyze_experiments import load, paired
from .audit_experiments import audit
from .data import agent_view, catalog, prepare, read_rows, taxonomy
from .diagnostic_campaign import ASR_RUNS
from .expanded_compare import merged, summary
from .extra_metrics import transcript_errors
from .full_compare import check_full_sources
from .full_selection import create
from .references import PATH as REFERENCE_PATH
from .scoring import compatible, exact
from .speech import audio_paths, audio_view


CONFIG_FIELDS = ["model", "split_sha256", "catalog_sha256", "runtime_py_sha256", "reference_sha256",
                 "prompt_sha256", "scorer", "scorer_assumptions", "runtime_tag", "seed", "thinking",
                 "temperature", "context", "max_new_tokens", "max_calls", "max_repairs", "deadline_s",
                 "normalization", "fine_tuned", "weight"]
FLAG_LABELS = {
    "wrong_or_extra_tool_name": "Có tên tool vượt số lần xuất hiện trong đáp án",
    "argument_not_matching_any_same_name_gold_call": "Có call đúng tên nhưng args không khớp call chuẩn nào cùng tên",
    "repeated_call_beyond_gold_multiplicity": "Lặp nguyên call nhiều hơn số lần trong đáp án",
    "finished_with_fewer_calls_than_gold": "Kết thúc bình thường nhưng số call ít hơn đáp án",
    "missing_required_ask": "Đáp án cần hỏi lại nhưng không gọi hỏi lại",
    "unexpected_ask": "Gọi hỏi lại khi đáp án không có hỏi lại",
    "id_argument_not_seen_in_visible_input": "Có ID chưa thấy trong input/observation trước call",
    "tool_returned_ok_false": "Tool trả ok=false",
    "tool_returned_done_false": "Tool trả done=false",
    "aligned_date_time_argument_mismatch": "Ngày/giờ khác đáp án tại vị trí call cùng tên",
    "call_budget_exhausted": "Chạm guard nội bộ 8 calls",
    "timeout": "Timeout",
    "processing_failure": "Lỗi xử lý khác",
}


def read_traces(paths):
    records = {}
    for directory in paths:
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        lines = [json.loads(s) for s in (directory / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
        if [r["id"] for r in lines] != manifest["ids"]:
            raise ValueError("Trace IDs differ from manifest: " + str(directory))
        for record in lines:
            if record["id"] in records:
                raise ValueError("Duplicate trace ID: " + record["id"])
            records[record["id"]] = record
    return records


def serialized(call):
    return json.dumps({"name": call["name"], "arguments": call["arguments"]}, ensure_ascii=False, sort_keys=True)


def flags_for(record, reference, view, tools):
    """Observable overlapping indicators, NOT a semantic/root-cause classifier."""
    predicted, gold = record["tool_calls"], reference["tool_calls"]
    flags = set()
    gold_names = Counter(c["name"] for c in gold)
    predicted_names = Counter(c["name"] for c in predicted)
    if predicted_names - gold_names:
        flags.add("wrong_or_extra_tool_name")
    if Counter(map(serialized, predicted)) - Counter(map(serialized, gold)):
        counts = Counter(map(serialized, predicted))
        expected = Counter(map(serialized, gold))
        if any(n > 1 and n > expected[call] for call, n in counts.items()):
            flags.add("repeated_call_beyond_gold_multiplicity")
    for call in predicted:
        same_name = [c for c in gold if c["name"] == call["name"]]
        if same_name and not any(compatible(call, c, tools) for c in same_name):
            flags.add("argument_not_matching_any_same_name_gold_call")
    if record["status"] == "finished" and len(predicted) < len(gold):
        flags.add("finished_with_fewer_calls_than_gold")
    ask = "sys.ask_missing_info"
    if gold_names[ask] and not predicted_names[ask]:
        flags.add("missing_required_ask")
    if predicted_names[ask] and not gold_names[ask]:
        flags.add("unexpected_ask")
    visible = json.dumps(initial_messages(view), ensure_ascii=False)
    for step in record["trace"]:
        if not step.get("executed"):
            continue
        for key, value in step["proposal"]["arguments"].items():
            if (key == "id" or key.endswith("_id")) and isinstance(value, str) and value not in visible:
                flags.add("id_argument_not_seen_in_visible_input")
        observation = step.get("observation", {})
        if isinstance(observation, dict):
            if observation.get("ok") is False:
                flags.add("tool_returned_ok_false")
            if observation.get("done") is False:
                flags.add("tool_returned_done_false")
        visible += json.dumps(observation, ensure_ascii=False)
    for p, g in zip(predicted, gold):
        if p["name"] != g["name"]:
            continue
        for key in set(p["arguments"]) & set(g["arguments"]):
            if re.search(r"(^|_)(date|time|start|end|until)($|_)", key) and not exact(p["arguments"][key], g["arguments"][key]):
                flags.add("aligned_date_time_argument_mismatch")
    if record["status"] == "timeout":
        flags.add("timeout")
    elif record["status"] == "call_budget_exhausted":
        flags.add("call_budget_exhausted")
    elif record["status"] not in ("finished", "asked_missing_info"):
        flags.add("processing_failure")
    prefix = 0
    for p, g in zip(predicted, gold):
        if not compatible(p, g, tools):
            break
        prefix += 1
    if prefix == len(predicted) == len(gold):
        divergence = "none"
    elif prefix >= len(predicted):
        divergence = "prediction_ended"
    elif prefix >= len(gold):
        divergence = "extra_call"
    else:
        divergence = "tool_name" if predicted[prefix]["name"] != gold[prefix]["name"] else "arguments"
    return {"flags": sorted(flags), "compatible_prefix_length": prefix, "first_sequence_divergence": divergence,
            "gold_calls_exceed_guard": len(gold) > 8}


def matrix(gold, asr, ids):
    result = Counter()
    for i in ids:
        g, a = gold["scores"][i]["score_spec_local"], asr["scores"][i]["score_spec_local"]
        result["both_positive" if g > 0 and a > 0 else "gold_positive_asr_zero" if g > 0 else "gold_zero_asr_positive" if a > 0 else "both_zero"] += 1
        result["gold_higher" if g > a + 1e-9 else "asr_higher" if a > g + 1e-9 else "equal"] += 1
    return dict(result)


def extra_summary(run, traces, references):
    m = summary(run)
    ids = run["manifest"]["ids"]
    m["strict_tool_f1"] = statistics.mean(run["scores"][i]["strict_tool_f1"] for i in ids)
    m["exact_sequence_accuracy"] = statistics.mean(exact(traces[i]["tool_calls"], references[i]["tool_calls"]) for i in ids)
    nonempty = [i for i in ids if references[i]["tool_calls"]]
    m["nonempty_gold_state_accuracy"] = statistics.mean(run["scores"][i]["state_match"] for i in nonempty)
    return m


def example_lines(i, rows, traces, runs, details, references):
    row = rows[i]
    transcript = traces["asr"][i]["asr"]["transcripts"][row["audio"]]
    lines = [f"### {i}", "", f"- Yêu cầu text chuẩn: {row['audio_text']}",
             f"- Phiên âm hiện tại: {transcript}",
             f"- History: {len(row.get('history', []))} turns; gold có {len(references[i]['tool_calls'])} calls.",
             f"- Điểm text chuẩn: {runs['gold']['scores'][i]['score_spec_local']:.2f}; audio: {runs['asr']['scores'][i]['score_spec_local']:.2f}.",
             "", "**Chuỗi chuẩn**", "", "```json", json.dumps(references[i]["tool_calls"], ensure_ascii=False, indent=2), "```"]
    for mode, label in [("gold", "Text chuẩn"), ("asr", "Audio")]:
        t, d = traces[mode][i], details[mode][i]
        lines += ["", f"**{label}: {t['status']} — {t['elapsed_s']:.2f}s**", "",
                  "Cờ: " + ("; ".join(FLAG_LABELS[f] for f in d["flags"]) or "Không có cờ heuristic"),
                  "", "```json", json.dumps(t["tool_calls"], ensure_ascii=False, indent=2), "```"]
        observations = [{"step": s["step"], "call": s.get("proposal"), "result": s.get("observation")}
                        for s in t["trace"] if s.get("executed")]
        lines += ["", "Kết quả tool thực tế:", "", "```json", json.dumps(observations, ensure_ascii=False, indent=2), "```"]
    return lines + [""]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection, split = create(), prepare()
    ids = selection["all_train"]
    gold = load(args.gold)
    audio_parts = [load(p) for p in ASR_RUNS]
    if gold["manifest"]["mode"] != "gold" or gold["manifest"]["partition"] != "all-train" or gold["manifest"]["ids"] != ids:
        raise ValueError("Expected the complete 1545-sample clean-text run")
    for run in [gold, *audio_parts]:
        for field in CONFIG_FIELDS:
            if run["manifest"].get(field) != gold["manifest"].get(field):
                raise ValueError("Uncontrolled difference: " + field)
    if gold["manifest"]["code_sha256"] != audio_parts[1]["manifest"]["code_sha256"]:
        raise ValueError("Inference code differs from the frozen full audio run")
    check_full_sources(args.gold)
    asr = merged(*audio_parts)
    if asr["manifest"]["ids"] != ids:
        raise ValueError("Audio selection differs from full clean-text selection")
    runs = {"gold": gold, "asr": asr}
    rows = {r["id"]: r for r in read_rows()}
    references = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))["references"]
    tools = catalog()
    traces = {"gold": read_traces([args.gold]), "asr": read_traces(ASR_RUNS)}
    details = {mode: {} for mode in runs}
    errors = {}
    for i in ids:
        audio = traces["asr"][i].get("asr")
        if not audio or set(audio["transcripts"]) != set(audio_paths(rows[i])):
            raise ValueError("Incomplete frozen ASR transcripts: " + i)
        for mode in runs:
            view = agent_view(rows[i]) if mode == "gold" else audio_view(rows[i], audio["transcripts"])
            details[mode][i] = flags_for(traces[mode][i], references[i], view, tools)
        errors[i] = transcript_errors(rows[i]["audio_text"], audio["transcripts"][rows[i]["audio"]])
    cohorts = {"full_1545": ids, "original_309": selection["original_309"],
               "without_12_smoke": [i for i in ids if i not in set(selection["smoke_included"])],
               "neither_timed_out": [i for i in ids if all(traces[m][i]["status"] != "timeout" for m in runs)]}
    comparisons = {name: paired({"manifest": {"ids": selected}, "scores": asr["scores"]},
                                {"manifest": {"ids": selected}, "scores": gold["scores"]}, split["group_ids"])
                   for name, selected in cohorts.items() if selected}
    metrics = {mode: extra_summary(run, traces[mode], references) for mode, run in runs.items()}
    flags = {mode: dict(Counter(f for i in ids for f in details[mode][i]["flags"])) for mode in runs}
    zero_flags = {mode: dict(Counter(f for i in ids if runs[mode]["scores"][i]["score_spec_local"] == 0
                                   for f in details[mode][i]["flags"])) for mode in runs}
    predicates = {"zero-call": lambda r: taxonomy(r)["calls"] == 0,
                  "single-call": lambda r: taxonomy(r)["calls"] == 1,
                  "multi-call>=3": lambda r: taxonomy(r)["calls"] >= 3,
                  "ask": lambda r: taxonomy(r)["ask"], "history": lambda r: taxonomy(r)["history_turns"] > 0,
                  "long-history>=10": lambda r: taxonomy(r)["history_turns"] >= 10}
    slices = {}
    for name, predicate in predicates.items():
        selected = [i for i in ids if predicate(rows[i])]
        slices[name] = {"n": len(selected), "scores": {mode: statistics.mean(run["scores"][i]["score_spec_local"] for i in selected)
                                                      if selected else None for mode, run in runs.items()}}
    verification = audit([args.gold, *ASR_RUNS])
    pair_matrix = matrix(gold, asr, ids)
    selected_cases = []
    categories = {
        "text_rescues_audio_zero": lambda i: gold["scores"][i]["score_spec_local"] > 0 and asr["scores"][i]["score_spec_local"] == 0,
        "both_zero": lambda i: gold["scores"][i]["score_spec_local"] == 0 and asr["scores"][i]["score_spec_local"] == 0,
        "audio_positive_text_zero": lambda i: gold["scores"][i]["score_spec_local"] == 0 and asr["scores"][i]["score_spec_local"] > 0,
    }
    case_groups = {}
    for name, predicate in categories.items():
        chosen = [i for i in ids if predicate(i)][:6]
        case_groups[name] = chosen
        selected_cases.extend(chosen)
    for flag in FLAG_LABELS:
        candidates = [i for i in ids if gold["scores"][i]["score_spec_local"] < 100
                      and flag in details["gold"][i]["flags"] and i not in selected_cases]
        selected_cases.extend(candidates[:2])
    args.output.mkdir(parents=True, exist_ok=False)
    analysis = {"metrics": metrics, "comparisons_gold_minus_asr": comparisons, "paired_outcomes": pair_matrix,
                "slices": slices, "flags_all_samples": flags, "flags_zero_score_samples": zero_flags,
                "flag_labels": FLAG_LABELS, "sample_diagnostics": details, "current_transcript_errors": errors,
                "audit": verification, "case_groups": case_groups, "selected_cases": selected_cases,
                "run_paths": {"gold": str(args.gold), "asr": list(map(str, ASR_RUNS))},
                "budget_stop_corrections": {"gold": gold["budget_stop_corrections"],
                                            "asr": [r["budget_stop_corrections"] for r in audio_parts]},
                "not_official_score": True, "causality_limit": "Changes all current/history transcripts and removes live ASR time; different hardware load/time. Does not uniquely isolate ASR recognition from LLM budget/runtime."}
    (args.output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (args.output / "metrics.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        fields = ["input", "samples", "score_100", "tool_f1_percent", "strict_f1_percent", "state_percent", "exact_sequence_percent",
                  "p50_total_s", "p95_total_s", "timeouts", "failures", "budget_stops"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode, m in metrics.items():
            writer.writerow(dict(zip(fields, [mode, m["n"], m["score"], 100*m["tool_f1"], 100*m["strict_tool_f1"],
                100*m["state_accuracy"], 100*m["exact_sequence_accuracy"], m["latency"]["p50_s"], m["latency"]["p95_s"],
                m["statuses"].get("timeout", 0), m["failures_counted_zero"], m["statuses"].get("call_budget_exhausted", 0)])))
    with (args.output / "per_sample.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "gold_score", "asr_score", "gold_minus_asr", "gold_status", "asr_status", "current_wer", "gold_flags", "asr_flags"])
        for i in ids:
            g, a = gold["scores"][i]["score_spec_local"], asr["scores"][i]["score_spec_local"]
            writer.writerow([i, g, a, g-a, traces["gold"][i]["status"], traces["asr"][i]["status"],
                             errors[i]["word_edits"]/errors[i]["reference_words"] if errors[i]["reference_words"] else "",
                             "|".join(details["gold"][i]["flags"]), "|".join(details["asr"][i]["flags"])])
    lines = ["# Chẩn đoán Qwen3.5-4B: text chuẩn so với audio — đủ 1.545 mẫu", "",
             "**Mục đích:** tìm hướng cải thiện trước fine-tune. Cùng Qwen3.5-4B Q4_K_M, prompt, agent, tools, scorer và giới hạn; chỉ lượt text chuẩn chạy mới. Tái sử dụng nguyên vẹn 309 + 1.236 kết quả audio đã có.",
             "**Đây là điểm local theo công thức BTC, chưa phải điểm leaderboard.** Không train, không upload, không loại/rerun chọn lọc mẫu lỗi.", "",
             "## Bảng metrics", "", "| Chỉ số | Text chuẩn | Audio qua ASR |", "|---|---:|---:|"]
    for label, function in [
        ("Số mẫu", lambda m: str(m["n"])), ("**Điểm chính /100**", lambda m: f"**{m['score']:.2f}**"),
        ("Tool F1", lambda m: f"{100*m['tool_f1']:.2f}%"), ("Strict Tool F1", lambda m: f"{100*m['strict_tool_f1']:.2f}%"),
        ("State hash đúng", lambda m: f"{100*m['state_accuracy']:.2f}%"), ("Đúng cả chuỗi call/args/thứ tự", lambda m: f"{100*m['exact_sequence_accuracy']:.2f}%"),
        ("p50 tổng", lambda m: f"{m['latency']['p50_s']:.2f}s"), ("p95 tổng", lambda m: f"{m['latency']['p95_s']:.2f}s"),
        ("Timeout", lambda m: str(m["statuses"].get("timeout", 0))), ("Tổng lỗi/timeout chấm 0", lambda m: str(m["failures_counted_zero"])),
        ("Chạm guard 8 calls", lambda m: str(m["statuses"].get("call_budget_exhausted", 0)))]:
        lines.append(f"| {label} | {function(metrics['gold'])} | {function(metrics['asr'])} |")
    delta = comparisons["full_1545"]
    lo, hi = delta["ci95_group_bootstrap"]
    lines += ["", f"Text chuẩn trừ audio: **{delta['delta_right_minus_left']:+.2f} điểm**; CI95% bootstrap theo group: **[{lo:+.2f}; {hi:+.2f}]** (10.000 lần).",
              f"Theo từng mẫu: text chuẩn cao hơn {delta['right_wins']}, thấp hơn {delta['right_losses']}, bằng {delta['ties']}.",
              "", "Điểm chính: gold có lệnh chỉ nhận Tool F1 khi state cuối đúng và có call khớp; sai state thì 0. Gold không có lệnh: pred rỗng được 100. Lỗi/timeout được 0. Tool F1 không tự áp state gate; state đúng cũng có thể chỉ do các lệnh đọc không đổi dữ liệu. p50 là trung vị, p95 là mốc 95% mẫu không vượt quá.",
              "", "## Chuyển đổi kết quả theo cùng ID", "", "| Kết quả | Số mẫu |", "|---|---:|",
              f"| Cả hai có điểm >0 | {pair_matrix.get('both_positive', 0)} |",
              f"| Text chuẩn >0, audio =0 | {pair_matrix.get('gold_positive_asr_zero', 0)} |",
              f"| Text chuẩn =0, audio >0 | {pair_matrix.get('gold_zero_asr_positive', 0)} |",
              f"| Cả hai =0 | {pair_matrix.get('both_zero', 0)} |",
              "", "Các nhóm >0 có thể vẫn chỉ đạt một phần điểm. Không gọi text chuẩn là trần năng lực hay số mẫu được cứu là toàn bộ lỗi do ASR.",
              "", "## Theo nhóm tác vụ", "", "| Nhóm | N | Text chuẩn /100 | Audio /100 |", "|---|---:|---:|---:|"]
    for name, value in slices.items():
        lines.append(f"| {name} | {value['n']} | {value['scores']['gold']:.2f} | {value['scores']['asr']:.2f} |")
    lines += ["", "## Dấu hiệu lỗi quan sát được", "",
              "Các cờ dưới đây được thống kê trên **toàn bộ mẫu**, có thể chồng lấp và có thể xuất hiện ở mẫu vẫn có điểm. Đây là heuristic để chọn trace cần đọc, không phải kết luận nguyên nhân hay lỗi ngữ nghĩa chắc chắn. Chuỗi khác đáp án vẫn có thể là cách làm hợp lệ; call thừa chủ yếu ảnh hưởng F1. ID chưa thấy chỉ là đối chiếu chuỗi với input/observations trước đó, không chứng minh model bịa ID. Ngày/giờ chỉ so tại vị trí call cùng tên, không suy luận tương đương ngữ nghĩa.",
              "", "| Dấu hiệu | Text chuẩn | Audio |", "|---|---:|---:|"]
    for flag, label in FLAG_LABELS.items():
        lines.append(f"| {label} | {flags['gold'].get(flag, 0)} | {flags['asr'].get(flag, 0)} |")
    over_guard = sum(len(references[i]["tool_calls"]) > 8 for i in ids)
    lines += ["", f"Đáp án có >8 calls: {over_guard} mẫu. Guard nội bộ là một giới hạn của pipeline, không phải giới hạn số tool calls do BTC công bố. Guard trong 30s được chấm theo calls/hash thực tế; correction lưu trong analysis.json, raw artifacts không sửa.",
              "", "## Các phép kiểm tra bổ sung", "", "| Bộ mẫu | N | Text trừ audio | CI95% |", "|---|---:|---:|---|"]
    for name, value in comparisons.items():
        a, b = value["ci95_group_bootstrap"]
        lines.append(f"| {name} | {value['samples']} | {value['delta_right_minus_left']:+.2f} | [{a:+.2f}; {b:+.2f}] |")
    lines += ["", "## Cách quyết định bước tiếp theo", "",
              "- Text chuẩn cải thiện rõ: bản phiên âm và khả năng chịu nhiễu là các hướng cần ưu tiên, nhưng phải đọc trace để biết sai từ nào làm đổi hành động.",
              "- Text chuẩn vẫn thấp: lỗi còn nằm trong LLM/agent/tool interaction, cách dùng history, điểm dừng hoặc các giới hạn hiện tại. Điều đó chưa chứng minh cần thay kiến trúc model.",
              "- Dùng ERROR_CASES.md và per_sample.csv để kiểm chứng các nhóm: text cứu audio, cả hai sai và audio tốt hơn text. Chỉ sửa pipeline khi tìm được lỗi cụ thể; thử SFT/LoRA có đối chứng sau đó. Không tự fine-tune trong lượt này.",
              "", "## Phạm vi, audit và giới hạn", "",
              "- Có đúng 1.545 ID duy nhất mỗi mode; cùng original train, gồm 12 smoke từng phát triển prompt. Vẫn có thể train sau này; đây không phải blind test cuối.",
              "- Gold thay cả transcript hiện tại **và user history**. Không thay gold tool observations vào input agent. Nhãn/reference chỉ dùng sau inference để chấm và phân tích.",
              "- Gold không tốn ASR nên có thêm thời gian dành cho LLM trong ngân sách 30s. Hai lượt khác thời điểm/tải/nhiệt trên RTX4060. Chênh điểm không cô lập hoàn toàn lỗi nhận dạng ASR; cohort không timeout chỉ là sensitivity check, không loại được mọi nhiễu.",
              "- Giữ Q4_K_M và non-thinking, không phải đo BF16/full-precision hoặc tác động quantization. Năng lực zero-shot không dự báo chắc mức cải thiện sau fine-tune.",
              "- Scorer local-spec-v1 có assumptions fuzzy/exact argument keys. State hash reference lấy từ replay TRAIN riêng; không đưa vào live agent. Cần đối chiếu portal khi nộp được phép.",
              f"- Audit replay hash/calls/token/reasoning/ASR coverage: **{'PASS' if verification['all_checks_passed'] else 'CÓ CỜ — xem analysis.json trước khi kết luận'}**.",
              "", "[Ví dụ trace lỗi](ERROR_CASES.md) · [CSV metrics](metrics.csv) · [Từng mẫu](per_sample.csv) · [Audit, CI và diagnostics](analysis.json)"]
    (args.output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    cases = ["# Ví dụ để đọc trace — chẩn đoán Qwen3.5-4B", "",
             "Chọn theo ID tăng dần: tối đa 6 mẫu mỗi nhóm đối chiếu và tối đa 2 mẫu bổ sung mỗi cờ ở lượt text chuẩn. Đây là minh họa có phân nhóm, không phải mẫu ngẫu nhiên đại diện tỷ lệ lỗi. Mọi mẫu vẫn có trong điểm tổng; không chạy lại hoặc loại mẫu.",
             "Các cờ là dấu hiệu tự động, chưa phải nhận xét ngữ nghĩa đã được con người xác nhận. Input history đầy đủ và response model nằm trong traces.jsonl gốc; dưới đây có yêu cầu hiện tại, calls chuẩn, calls thực tế và observations để kiểm tra.", ""]
    for name, chosen in case_groups.items():
        cases.append(f"- {name}: {', '.join(chosen) or 'không có'}")
    cases.append("")
    for i in selected_cases:
        cases += example_lines(i, rows, traces, runs, details, references)
    (args.output / "ERROR_CASES.md").write_text("\n".join(cases)+"\n", encoding="utf-8")
    print("DIAGNOSTIC_REPORT", args.output / "REPORT.md", flush=True)
    print(json.dumps({"metrics": metrics, "gold_minus_asr": delta, "outcomes": pair_matrix,
                      "audit_passed": verification["all_checks_passed"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
