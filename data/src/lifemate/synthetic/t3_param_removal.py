"""T3 — lược 1 tham số khỏi câu nói -> gold kết thúc bằng sys.ask_missing_info.

Sửa so với bản cũ:
  * Bản cũ: re.sub xóa MỌI con số khớp regex (xóa luôn ngưỡng giá vàng "131 triệu"),
    regex giờ để sót "kém 15", không bắt số viết bằng chữ ("hai triệu"),
    không kiểm tra con số bị xóa thuộc lệnh nào (xóa "50 nghìn" của topup nhưng hỏi amount của transfer),
    và gán cứng final_state = {}.
  * Bản mới:
      - chỉ xóa ĐÚNG MỘT cụm có giá trị bằng giá trị gold của tham số mục tiêu (viet_spans);
      - số tiền -> thay bằng "chút tiền"/"ít tiền" (kiểu câu train: "chuyển ví cho bố chút tiền");
        giờ -> xóa phần giờ, giữ buổi/ngày (train: "nhắc anh uống liều Cardiol sáng mai" vẫn hỏi time);
      - bỏ mẫu có tự sửa ("à không", "à nhầm"...), mẫu mà giá trị còn dùng ở lệnh khác,
        mẫu mà giá trị xuất hiện trong history, mẫu còn sót cụm giờ khác không thuộc lệnh nào;
      - prefix (các lệnh gold trước lệnh ghi cuối) được replay thật; final_state = state các app
        bị lệnh ghi trong prefix thay đổi (quy ước train: 79/79 mẫu "ghi trước ask" có final_state).
"""
from __future__ import annotations

import re

from .common import norm_text, tool_map
from .viet_spans import cleanup, find_amount_spans, find_time_spans, has_self_correction

AMOUNT_PARAMS = {"amount"}
TIME_PARAMS = {"time", "start"}
ISO_RE = re.compile(r"^\d{4}-\d\d-\d\dT(\d\d):(\d\d)")


def removable_params(call: dict, schema: dict) -> list[str]:
    props = schema.get("parameters", {}).get("properties", {})
    req = schema.get("parameters", {}).get("required", [])
    out = []
    for p in req:
        if p not in call["arguments"]:
            continue
        sp = props.get(p, {})
        if "enum" in sp or "x-ref" in sp:
            continue
        v = call["arguments"][p]
        if p in AMOUNT_PARAMS and isinstance(v, int) and v >= 1000:
            out.append(p)
        elif p in TIME_PARAMS and isinstance(v, str) and ISO_RE.match(v):
            out.append(p)
    return out


def _hm(v: str):
    m = ISO_RE.match(v)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _other_values(calls, w, param):
    """Giá trị cùng loại đang được DÙNG ở các lệnh khác (và ở các tham số khác của w)."""
    amounts, times = [], []
    for c in calls:
        for k, v in c["arguments"].items():
            if c is w and k == param:
                continue
            if isinstance(v, int) and v >= 1000:
                amounts.append(v)
            if isinstance(v, str) and ISO_RE.match(v):
                times.append(_hm(v))
    return amounts, times


def _in_other_free_text(calls, w, snippet: str, extra_forms=()) -> bool:
    """Cụm bị xóa có nằm trong nội dung chữ (content/body/note...) của lệnh khác không."""
    forms = [norm_text(snippet)] + [norm_text(f) for f in extra_forms]
    for c in calls:
        if c is w:
            continue
        for v in c["arguments"].values():
            if isinstance(v, str) and not ISO_RE.match(v):
                nv = norm_text(v)
                if any(f and f in nv for f in forms):
                    return True
    return False


def rewrite_text(sample: dict, w: dict, param: str, rng) -> tuple[str | None, str]:
    """Trả (câu_mới, lý_do). câu_mới=None nghĩa là bỏ mẫu."""
    text = sample["audio_text"]
    if has_self_correction(text):
        return None, "self_correction"
    calls = sample["label"]["tool_calls"]
    v = w["arguments"][param]
    other_amt, other_time = _other_values(calls, w, param)
    hist_text = " ".join(h.get("audio_text") or h.get("text") or "" for h in sample.get("history", []))

    if param in AMOUNT_PARAMS:
        if v in other_amt:
            return None, "value_used_by_other_call"
        if any(x == v for *_, x in find_amount_spans(hist_text)):
            return None, "value_in_history"
        spans = [s for s in find_amount_spans(text) if s[2] == v]
        if len(spans) != 1:
            return None, f"amount_spans={len(spans)}"
        s, e, _ = spans[0]
        if _in_other_free_text(calls, w, text[s:e], (f"{v:,}".replace(",", "."), str(v))):
            return None, "value_in_other_call_text"
        nxt = text[e:].lstrip().lower()
        filler = rng.choice(["chút", "ít"]) if nxt.startswith("tiền") else rng.choice(["chút tiền", "ít tiền"])
        new = cleanup(text[:s] + filler + text[e:])
        if any(x == v for *_, x in find_amount_spans(new)):
            return None, "value_still_present"
        return new, "ok"

    if param in TIME_PARAMS:
        hm = _hm(v)
        if hm in other_time:
            return None, "value_used_by_other_call"
        if any(x[2] == hm for x in find_time_spans(hist_text)):
            return None, "value_in_history"
        all_spans = find_time_spans(text)
        spans = [x for x in all_spans if x[2] == hm]
        if len(spans) != 1:
            return None, f"time_spans={len(spans)}"
        leftovers = [x for x in all_spans if x is not spans[0] and x[2] not in other_time]
        if leftovers:
            return None, "other_unexplained_time_in_text"
        _, _, _, cs, ce = spans[0]
        h, m = hm
        h12 = h - 12 if h > 12 else h
        if _in_other_free_text(calls, w, text[cs:ce], (f"{h}:{m:02d}", f"{h12}:{m:02d}", f"{h}h", f"{h12}h",
                                                       f"{h} giờ", f"{h12} giờ")):
            return None, "value_in_other_call_text"
        new = cleanup(text[:cs] + text[ce:])
        if any(x[2] == hm for x in find_time_spans(new)) or len(new) < 8:
            return None, "value_still_present_or_too_short"
        return new, "ok"
    return None, "unsupported_param"


def pick_target(sample, tmap):
    """Lệnh ghi CUỐI cùng của gold; không lấy mẫu đã có sys.* trong gold."""
    calls = sample["label"]["tool_calls"]
    if not calls or any(c["name"].startswith("sys.") for c in calls):
        return None
    idx = max((i for i, c in enumerate(calls) if tmap[c["name"]]["kind"] == "write"), default=None)
    return None if idx is None else idx


def plan_t3(samples, catalog, rng):
    """Phần KHÔNG cần ToolEnv: chọn mẫu + viết lại câu. Trả list kế hoạch + thống kê lý do loại."""
    from collections import Counter
    tmap = tool_map(catalog)
    plans, why = [], Counter()
    for s in samples:
        idx = pick_target(s, tmap)
        if idx is None:
            why["no_target_write"] += 1
            continue
        w = s["label"]["tool_calls"][idx]
        params = removable_params(w, tmap[w["name"]])
        if not params:
            why["no_removable_param"] += 1
            continue
        param = rng.choice(sorted(params))
        new, reason = rewrite_text(s, w, param, rng)
        why[reason] += 1
        if new:
            plans.append({"sample": s, "w_idx": idx, "param": param, "new_text": new})
    return plans, why


def build_t3(b, samples, catalog, target=375, llm=None):
    """Phần cần ToolEnv: replay prefix trên state synthetic, tính final_state, thêm record."""
    from .env_runner import final_state_of, replay, session_from, train_state
    if llm is not None:
        raise SystemExit("t3 --mode llm đã bỏ: chưa có cách kiểm tra tự động câu LLM viết lại. Dùng --mode rule.")
    plans, why = plan_t3(samples, catalog, b.rng)
    b.rng.shuffle(plans)
    made = 0
    for p in plans:
        if made >= target:
            break
        s, idx, param = p["sample"], p["w_idx"], p["param"]
        calls = s["label"]["tool_calls"]
        w, prefix = calls[idx], calls[:idx]
        states = train_state(s["id"])
        env = session_from(s, states, "probe")
        before = env.get_state()
        obs = replay(env, prefix)
        if obs is None:
            why["prefix_replay_fail"] += 1
            continue
        ask = {"name": "sys.ask_missing_info", "arguments": {"tool_name": w["name"], "params": [param]}}
        ra = env.call(ask["name"], ask["arguments"])
        fs = final_state_of(env, before=before)
        b.assert_source(s["id"])
        sid = b.new_id()
        b.write_state(sid, s, states)
        b.add({"id": sid, "source_id": s["id"], "technique": "t3_param_removal", "audio": s["audio"],
               "audio_source": f"train:{s['id']}", "transcript_mode": "rule_rewrite",
               "audio_text": p["new_text"], "context": s["context"], "history": s.get("history", []),
               "label": {"tool_calls": prefix + [ask], "tool_results": obs + [ra], "final_state": fs},
               "provenance": {"removed_param": param, "removed_value": w["arguments"][param],
                              "target_tool": w["name"], "original_text": s["audio_text"]}})
        made += 1
    print("t3 lý do loại / giữ:", dict(why))
    return made
