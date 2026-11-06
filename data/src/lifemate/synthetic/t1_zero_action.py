"""T1 — gỡ khỏi installed_apps mọi app mà gold dùng -> không app nào xử lý được -> gold = [].

Sửa so với bản cũ:
  * Guard "app thay thế" khác group: bản cũ chỉ chặn app CÙNG group, nên lọt 3 mẫu gỡ
    shopping_list nhưng vẫn còn notes/tasks (vẫn "ghi vào danh sách mua" được) -> nhãn [] đáng ngờ.
  * Kiểm chứng: mọi call gold đều phải bị ToolEnv từ chối khi dùng installed_apps mới
    (chạy được qua Simulator trong bộ nhớ, không ghi state).
"""
from copy import deepcopy

from .common import app_group, gold_ns, tool_map

# app khác group nhưng vẫn làm được việc tương tự
CROSS_GROUP_SUBSTITUTES = {
    "shopping_list": {"notes", "tasks", "reminders"},
    "notes": {"shopping_list"},
    "tasks": {"shopping_list"},
    "calculator": set(),
}


def hist_uses(s, ns):
    for t in s.get("history", []):
        for c in t.get("tool_calls", []) or []:
            if c["name"].split(".")[0] in ns:
                return True
    return False


def _all_rejected(sample, ctx, catalog):
    """Mọi call gold phải bị từ chối khi installed_apps = ctx (đúng luật: tool không thuộc app đã cài)."""
    from .env_runner import _zaic_tools
    zt = _zaic_tools()
    sim = zt.executor.Simulator({"id": "t1", "context": ctx}, catalog)
    return all(not sim.validate(c["name"], c["arguments"])["ok"] for c in sample["label"]["tool_calls"])


def build_t1(b, samples, catalog, target=500):
    groups = app_group(catalog)
    tool_map(catalog)
    cands = [s for s in samples if s["label"]["tool_calls"]
             and not any(c["name"].startswith("sys.") for c in s["label"]["tool_calls"])]
    b.rng.shuffle(cands)
    made, why = 0, {}

    def skip(k):
        why[k] = why.get(k, 0) + 1

    for s in cands:
        if made >= target:
            break
        ns = gold_ns(s)
        inst = s["context"]["installed_apps"]
        if not ns.issubset(inst):
            skip("gold_app_not_installed"); continue
        if hist_uses(s, ns):
            skip("history_uses_app"); continue
        remaining = {a for a in inst if a not in ns}
        if {groups[a] for a in remaining} & {groups[a] for a in ns}:
            skip("same_group_app_left"); continue
        if any(CROSS_GROUP_SUBSTITUTES.get(a, set()) & remaining for a in ns):
            skip("substitute_app_left"); continue
        ctx = deepcopy(s["context"])
        for a in ns:
            ctx["installed_apps"].pop(a, None)
        if not _all_rejected(s, ctx, catalog):
            skip("gold_call_still_allowed"); continue
        b.assert_source(s["id"])
        sid = b.new_id()
        b.add({"id": sid, "source_id": s["id"], "technique": "t1_zero_action", "audio": s["audio"],
               "audio_source": f"train:{s['id']}", "transcript_mode": "gold", "audio_text": s["audio_text"],
               "context": ctx, "history": s.get("history", []),
               "label": {"tool_calls": [], "tool_results": [], "final_state": {}},
               "provenance": {"removed_apps": sorted(ns)}})
        made += 1
    print("t1 lý do loại:", why)
    return made
