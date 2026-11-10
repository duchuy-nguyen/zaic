"""Kiểm định Phase 1 — bản v2.

Bản cũ chỉ kiểm tra: hình dạng nhãn (và còn ÉP t2/t3 phải có final_state == {}), trùng transcript VAL,
trùng lặp, và smoke replay "ok == true" trên một mẫu ngẫu nhiên -> không phát hiện được nhãn sai.

Bản v2 (mọi lỗi đều in id; exit code 1 nếu có lỗi):
  C1 structure   hình dạng nhãn theo kỹ thuật; ask là lệnh cuối và duy nhất.
  C2 schema      mọi call hợp lệ theo luật zaic_tools với installed_apps của CHÍNH record.
  C3 no_syn      không còn chuỗi "_syn" ở bất kỳ đâu.
  C4 phone       số điện thoại trong tool_results dạng ^0\\d{9,10}$.
  C5 leak        source thuộc TRAIN_FIT mới; transcript không trùng câu nào (kể cả history) của VAL.
  C6 dup         không trùng (audio_text + context).
  C7 replay      replay TOÀN BỘ record trên state synthetic: tool_results phải BẰNG nhãn,
                 final_state phải BẰNG các app bị đổi (định nghĩa Executor.replay của zaic_tools).
  C8 provenance  state synthetic khớp nguồn: t3/t4 cùng hash với mẫu train gốc; t2 = gốc + đúng 1 thực thể mới.
  C10 alignment  t1/t2: audio_text == câu gốc (WAV khớp text); t3/t4: khác câu gốc và transcript_mode != gold
                 (chỉ dùng train text). Record mang cờ text_only để bên huấn luyện lọc.
  C11 t2_natural bản sao t2 phải khác bản gốc ngoài id (không còn bản sao chỉ-khác-ID); tên liên hệ là tên riêng.
  C9 semantic    t1: mọi call gold gốc bị từ chối với installed_apps mới.
                 t2: kết quả read cuối có >= 2 thực thể trùng tên, thực thể mới khác id gốc.
                 t3: giá trị đã lược không còn trong câu; câu khác câu gốc.
                 t4: câu nhiễu khác câu gốc đúng 1 token; nhãn = gold gốc.
Chạy (từ LifeMate/):
  python -m src.lifemate.synthetic.validate_phase1                  # các file synthetic/phase1_*.jsonl
  python -m src.lifemate.synthetic.validate_phase1 --file synthetic/synthetic_phase1.jsonl
  python -m src.lifemate.synthetic.validate_phase1 --file <cũ>.jsonl --no-replay   # khi không có state synthetic
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter, defaultdict

from .common import (OFFICIAL, PHONE_RE, SYN_ROOT, load_catalog, load_fit_ids, load_jsonl, load_val_ids,
                     norm_text, tool_map, tokens)

TECHS = ["t1", "t2", "t3", "t4c"]


def _phones(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("phone", "phone_number") and isinstance(v, str):
                acc.append(v)
            else:
                _phones(v, acc)
    elif isinstance(o, list):
        for v in o:
            _phones(v, acc)
    return acc


def _entities_by_name(results):
    """{(list_key, name_norm): [ids]} trên mọi list thực thể trong kết quả."""
    out = defaultdict(list)
    for r in results:
        stack = [(None, r)]
        while stack:
            key, o = stack.pop()
            if isinstance(o, dict):
                for k, v in o.items():
                    stack.append((k, v))
            elif isinstance(o, list):
                for x in o:
                    if isinstance(x, dict) and "id" in x:
                        n = x.get("name") or x.get("title")
                        if isinstance(n, str):
                            out[(key, norm_text(n))].append(x["id"])
                    stack.append((key, x))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--no-replay", action="store_true", help="bỏ C7/C8 (khi không có synthetic_environment)")
    ap.add_argument("--json-out", default=None, help="ghi kết quả kiểm định ra file JSON (bằng chứng cho OPS)")
    a = ap.parse_args()

    if a.file:
        recs = load_jsonl(a.file)
    else:
        recs = [r for t in TECHS if (SYN_ROOT / f"phase1_{t}.jsonl").exists() for r in load_jsonl(SYN_ROOT / f"phase1_{t}.jsonl")]
    cat = load_catalog("train")
    tmap = tool_map(cat)
    fit, val = load_fit_ids(), load_val_ids()
    train = {s["id"]: s for s in load_jsonl(OFFICIAL / "train" / "train.jsonl")}
    val_text = set()
    for i in val:
        s = train[i]
        val_text.add(norm_text(s["audio_text"]))
        for h in s.get("history", []):
            if h.get("role") == "user":
                val_text.add(norm_text(h.get("audio_text")))

    from .env_runner import _zaic_tools
    zt = _zaic_tools()
    errs = defaultdict(list)
    seen = set()

    for r in recs:
        rid, tech, L = r["id"], r["technique"], r["label"]
        calls, results = L["tool_calls"], L["tool_results"]
        src = train.get(r["source_id"])
        # C1
        if tech == "t1_zero_action":
            if calls or results or L["final_state"]:
                errs["C1_structure"].append(rid)
        else:
            if len(calls) != len(results):
                errs["C1_structure"].append(rid)
            if tech in ("t2_state_mutation", "t3_param_removal"):
                asks = [i for i, c in enumerate(calls) if c["name"] == "sys.ask_missing_info"]
                if len(asks) != 1 or asks[0] != len(calls) - 1:
                    errs["C1_structure"].append(rid)
        # C2
        sim = zt.executor.Simulator({"id": rid, "context": r["context"]}, cat)
        for c in calls:
            v = sim.validate(c["name"], c["arguments"])
            if not v["ok"]:
                errs["C2_schema"].append(f"{rid}:{c['name']}:{v['error']}")
        # C3, C4
        if "_syn" in json.dumps(r, ensure_ascii=False):
            errs["C3_no_syn"].append(rid)
        bad_ph = [p for p in _phones(results, []) if not PHONE_RE.match(p)]
        if bad_ph:
            errs["C4_phone"].append(f"{rid}:{bad_ph[:2]}")
        # C5
        if r["source_id"] not in fit or r["source_id"] in val:
            errs["C5_leak_source"].append(rid)
        if norm_text(r["audio_text"]) in val_text:
            errs["C5_leak_text"].append(rid)
        # C6
        h = (norm_text(r["audio_text"]), json.dumps(r["context"], sort_keys=True, ensure_ascii=False))
        if h in seen:
            errs["C6_dup"].append(rid)
        seen.add(h)
        # C10
        if src:
            same = r["audio_text"] == src["audio_text"]
            if tech in ("t1_zero_action", "t2_state_mutation") and (not same or r["audio"] != src["audio"]):
                errs["C10_audio_text_misaligned"].append(rid)
            if tech in ("t3_param_removal", "t4_asr_noise") and (same or r.get("transcript_mode") == "gold"):
                errs["C10_text_only_flag_wrong"].append(rid)
        # C11
        if tech == "t2_state_mutation":
            ch = r["provenance"].get("changed", {})
            if set(ch) <= {"id"}:
                errs["C11_t2_id_only_clone"].append(rid)
            lst = r["provenance"].get("entity_list") or \
                ("contacts" if r["provenance"].get("param") == "contact_id" else None)
            if lst == "contacts":
                from .mutators import is_person_name
                if not is_person_name(r["provenance"].get("duplicate_name", "")):
                    errs["C11_t2_not_person_name"].append(rid)
        # C9
        if tech == "t1_zero_action" and src:
            if any(sim.validate(c["name"], c["arguments"])["ok"] for c in src["label"]["tool_calls"]):
                errs["C9_t1_gold_still_callable"].append(rid)
        if tech == "t2_state_mutation":
            groups = _entities_by_name(results[:-1])
            if not any(len(set(ids)) >= 2 for ids in groups.values()):
                errs["C9_t2_no_ambiguity_in_results"].append(rid)
        if tech == "t3_param_removal":
            from .viet_spans import find_amount_spans, find_time_spans
            v = r["provenance"].get("removed_value")
            p = r["provenance"].get("removed_param")
            if v is None:
                errs["C9_t3_missing_provenance"].append(rid)
            elif p == "amount" and any(x == v for *_, x in find_amount_spans(r["audio_text"])):
                errs["C9_t3_value_still_in_text"].append(rid)
            elif p in ("time", "start") and isinstance(v, str) and \
                    any(x[2] == (int(v[11:13]), int(v[14:16])) for x in find_time_spans(r["audio_text"])):
                errs["C9_t3_value_still_in_text"].append(rid)
            if norm_text(r["audio_text"]) == norm_text(r["provenance"].get("original_text")):
                errs["C9_t3_text_unchanged"].append(rid)
        if tech == "t4_asr_noise":
            o = r["provenance"].get("original_text")
            if not o or len(tokens(o)) != len(tokens(r["audio_text"])) or \
                    sum(x != y for x, y in zip(tokens(o), tokens(r["audio_text"]))) != 1:
                errs["C9_t4_not_one_token"].append(rid)
            if src and calls != src["label"]["tool_calls"]:
                errs["C9_t4_label_not_gold"].append(rid)

        # C7, C8
        if a.no_replay or tech == "t1_zero_action":
            continue
        try:
            env = zt.ToolEnv("synthetic", rid, root=SYN_ROOT.parent)
        except FileNotFoundError:
            errs["C7_no_state"].append(rid)
            continue
        before = env.get_state()
        got = [env.call(c["name"], c["arguments"]) for c in calls]
        if got != results:
            errs["C7_results_differ"].append(rid)
        changed = {k: v for k, v in env.get_state().items() if v != before.get(k)}
        if changed != L["final_state"]:
            errs["C7_final_state_differ"].append(rid)
        if src:
            orig = zt.ToolEnv("train", src["id"], root=OFFICIAL)
            if tech in ("t3_param_removal", "t4_asr_noise"):
                if zt.ToolEnv("synthetic", rid, root=SYN_ROOT.parent).get_state_hash() != orig.get_state_hash():
                    errs["C8_state_not_source"].append(rid)
            else:
                s0, s1 = orig.get_state(), before
                d0, d1 = json.dumps(s0, ensure_ascii=False, sort_keys=True), json.dumps(s1, ensure_ascii=False, sort_keys=True)
                n_ids0 = len(re.findall(r'"id":', d0))
                n_ids1 = len(re.findall(r'"id":', d1))
                if n_ids1 != n_ids0 + 1:
                    errs["C8_t2_not_one_added_entity"].append(rid)

    by = Counter(r["technique"] for r in recs)
    print(f"records: {len(recs)} {dict(by)}")
    n_err = 0
    for k in sorted(errs):
        print(f"  {k}: {len(errs[k])}  e.g. {errs[k][:4]}")
        n_err += len(errs[k])
    if not n_err:
        print("  (không có lỗi)")
    print("VALIDATE OK" if not n_err else f"VALIDATE FAIL ({n_err} lỗi)")
    if a.json_out:
        import hashlib
        src_file = a.file or "synthetic/phase1_*.jsonl"
        sha = hashlib.sha256(open(a.file, "rb").read()).hexdigest() if a.file else None
        rep = {"file": src_file, "sha256": sha, "records": len(recs), "by_technique": dict(by),
               "replay": "skipped" if a.no_replay else "all_records_with_calls",
               "checks": ["C1_structure", "C2_schema", "C3_no_syn", "C4_phone", "C5_leak_source_and_VAL_text",
                          "C6_dup", "C7_replay_results_and_final_state", "C8_state_provenance_vs_train_source",
                          "C9_semantic_rules", "C10_audio_text_alignment", "C11_t2_natural_clone"],
               "errors": {k: v for k, v in errs.items()}, "error_count": n_err,
               "text_only_records": sum(r["technique"] in ("t3_param_removal", "t4_asr_noise") for r in recs),
               "not_checked": ["full_independent_semantic_adjudication (xem audit_phase1.csv)",
                               "training_quality_on_VAL"]}
        open(a.json_out, "w", encoding="utf-8", newline="\n").write(json.dumps(rep, ensure_ascii=False, indent=1))
        print("->", a.json_out)
    sys.exit(0 if not n_err else 1)


if __name__ == "__main__":
    main()
