"""T2 — thêm thực thể trùng tên vào state -> lệnh ghi cuối thành mơ hồ -> gold kết thúc bằng ask.

Sửa so với bản cũ:
  * Chỉ sinh cho cặp (tool, tham số x-ref) ĐÃ CÓ trong các mẫu ask của TRAIN_FIT
    (train: phone.call/sms.send.contact_id chiếm phần lớn; smart_home.light_set không có mẫu nào,
    bản cũ lại có 56 mẫu smart_home). Phân bổ số mẫu theo tỉ lệ train.
  * Nhắm lệnh ghi CUỐI cùng của gold (bản cũ lấy writes[0] -> 137/259 mẫu bỏ mất hành động phía sau,
    train không có mẫu nào như vậy: ask luôn là lệnh cuối, phần mơ hồ là việc cuối).
  * Replay TẤT CẢ lệnh gold đứng trước lệnh đó (cả read lẫn write), đúng quy ước train
    "làm phần rõ ràng trước rồi mới hỏi"; final_state = các app bị thay đổi (bản cũ gán cứng {}).
  * ID/điện thoại/thuộc tính bản sao theo kiểu train (mutators.py), không còn "_syn".
  * State gốc lấy qua ToolEnv.get_state() (không đọc thẳng state.json).
"""
from __future__ import annotations

import json
from collections import Counter

from .common import tool_map, xref_params
from .env_runner import final_state_of, replay, session_from, train_state
from .mutators import duplicate_for_ambiguity


def allowed_pairs(samples, tmap) -> Counter:
    """(tool_name, param) của các mẫu ask vì thực thể mơ hồ trong TRAIN_FIT."""
    c = Counter()
    for s in samples:
        calls = s["label"]["tool_calls"]
        if calls and calls[-1]["name"] == "sys.ask_missing_info":
            a = calls[-1]["arguments"]
            props = tmap.get(a.get("tool_name"), {}).get("parameters", {}).get("properties", {})
            for p in a.get("params", []):
                if "x-ref" in props.get(p, {}):
                    c[(a["tool_name"], p)] += 1
    return c


NEUTRAL_RELATIONS = {"đồng nghiệp", "khách hàng", "bạn", "hàng xóm", "bạn cùng lớp", "đối tác", "bạn thân"}


def relation_pool(samples):
    """Quan hệ thay thế: chỉ lấy loại trung tính về giới/tuổi và có thật trong train
    (tránh 'Tuấn - vợ'); train dùng kiểu 'Minh - đồng nghiệp' vs 'Minh - khách hàng'."""
    rel = Counter()
    for s in samples:
        for res in s["label"]["tool_results"]:
            for x in res.get("contacts", []) if isinstance(res, dict) else []:
                if isinstance(x, dict) and isinstance(x.get("relation"), str):
                    rel[x["relation"]] += 1
    return [r for r, n in rel.most_common() if n >= 3 and r in NEUTRAL_RELATIONS]


def build_t2(b, samples, catalog, target=150):
    tmap = tool_map(catalog)
    pairs = allowed_pairs(samples, tmap)
    # cùng loại thực thể với cặp đã có trong train (chat_ott.chats) -> cho phép thêm
    for extra in (("chat_ott.send_location", "chat_id"), ("chat_ott.call", "chat_id")):
        pairs.setdefault(extra, 1)
    total = sum(pairs.values())
    quota = {k: max(1, round(target * v / total)) for k, v in pairs.items()}
    rels = relation_pool(samples)
    print("t2 cặp (tool,param) cho phép:", dict(pairs.most_common()))
    cands = [s for s in samples if s["label"]["tool_calls"]
             and not any(c["name"].startswith("sys.") for c in s["label"]["tool_calls"])]
    b.rng.shuffle(cands)
    made, used, why = 0, Counter(), Counter()
    for s in cands:
        if made >= target:
            break
        calls = s["label"]["tool_calls"]
        widx = max((i for i, c in enumerate(calls) if tmap[c["name"]]["kind"] == "write"), default=None)
        if widx is None:
            why["no_write"] += 1; continue
        w = calls[widx]
        ps = [p for p in xref_params(tmap[w["name"]]) if (w["name"], p) in pairs and w["arguments"].get(p)]
        if not ps:
            why["pair_not_in_train"] += 1; continue
        p = ps[0]
        if used[(w["name"], p)] >= quota[(w["name"], p)]:
            why["quota_full"] += 1; continue
        if not any(tmap[c["name"]]["kind"] == "read" for c in calls[:widx]):
            why["no_read_before"] += 1; continue
        rid = w["arguments"][p]
        if rid in json.dumps(s.get("history", []), ensure_ascii=False):
            why["entity_in_history"] += 1; continue
        if any(rid in json.dumps(c["arguments"], ensure_ascii=False) for c in calls if c is not w):
            why["entity_used_by_other_call"] += 1; continue
        states = train_state(s["id"])
        info = duplicate_for_ambiguity(states, rid, s["audio_text"], rels, b.rng)
        if not info:
            why["cannot_duplicate_naturally"] += 1; continue
        if "skip" in info:
            why[info["skip"]] += 1; continue
        env = session_from(s, states, "probe")
        before = env.get_state()
        prefix = calls[:widx]
        obs = replay(env, prefix)
        if obs is None:
            why["prefix_replay_fail"] += 1; continue
        if not any(rid in json.dumps(o, ensure_ascii=False) and info["new_id"] in json.dumps(o, ensure_ascii=False)
                   for o in obs):
            why["read_does_not_reveal_both"] += 1; continue
        ask = {"name": "sys.ask_missing_info", "arguments": {"tool_name": w["name"], "params": [p]}}
        ra = env.call(ask["name"], ask["arguments"])
        fs = final_state_of(env, before=before)
        b.assert_source(s["id"])
        sid = b.new_id()
        b.write_state(sid, s, states)
        b.add({"id": sid, "source_id": s["id"], "technique": "t2_state_mutation", "audio": s["audio"],
               "audio_source": f"train:{s['id']}", "transcript_mode": "gold", "audio_text": s["audio_text"],
               "context": s["context"], "history": s.get("history", []),
               "label": {"tool_calls": prefix + [ask], "tool_results": obs + [ra], "final_state": fs},
               "provenance": {"target_tool": w["name"], "param": p, "duplicated_from": rid,
                              "duplicate_name": info["name"], "changed": info["changed"], "entity_list": info.get("list"),
                              "dropped_calls_after": len(calls) - widx - 1}})
        used[(w["name"], p)] += 1
        made += 1
    print("t2 lý do loại:", dict(why))
    print("t2 theo cặp:", dict(used.most_common()))
    return made
