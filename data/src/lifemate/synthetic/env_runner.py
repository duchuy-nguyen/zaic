"""Cầu nối ToolEnv cho pha sinh data (chỉ chạy offline trên TRAIN, không import vào agent).

Sửa so với bản cũ:
  * KHÔNG đọc/copy trực tiếp train_environment/<id>/state.json nữa. State gốc lấy qua
    ToolEnv("train", id).get_state() — API mà README zaic_tools cho phép dùng trên train
    ("để debug hoặc so với label.final_state của mẫu train").
  * `_meta` của state synthetic được dựng lại từ context của mẫu (current_time, installed tools, city),
    không chép từ file gốc. check_roundtrip() kiểm chứng cách dựng này cho ra đúng hash + đúng nhãn.
  * final_state_of(): đúng định nghĩa Executor.replay() của zaic_tools = state của các app bị thay đổi.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from .common import OFFICIAL, SYN_ROOT  # noqa: F401  (common cũng cấu hình stdout UTF-8)

_ZT = None


def _zaic_tools():
    """Import zaic_tools một lần. Bản trong synthetic/ là bản copy y nguyên của train."""
    global _ZT
    if _ZT is None:
        for p in (SYN_ROOT, OFFICIAL / "train"):
            if (p / "zaic_tools").is_dir():
                sp = str(p.resolve())
                if sp not in sys.path:
                    sys.path.insert(0, sp)
                break
        import zaic_tools  # noqa: E402
        _ZT = zaic_tools
    return _ZT


def train_env(sid: str):
    return _zaic_tools().ToolEnv("train", sid, root=OFFICIAL)


def train_state(sid: str) -> dict:
    """State ban đầu (đã áp history) của mẫu train, qua API được phép."""
    return train_env(sid).get_state()


def build_meta(sample: dict, qid: str) -> dict:
    ctx = sample["context"]
    tools = sorted(t for a in ctx["installed_apps"].values() for t in a.get("tools", []))
    return {"question_id": qid, "current_time": ctx["current_time"],
            "city": (ctx.get("user_profile") or {}).get("city"), "tools": tools}


def make_env(sid: str):
    """Phiên ToolEnv mới trên synthetic/synthetic_environment/<sid>/state.json."""
    return _zaic_tools().ToolEnv("synthetic", sid, root=SYN_ROOT.parent if SYN_ROOT.parent != Path("") else Path("."))


def session_from(sample: dict, states: dict, qid: str = "tmp"):
    """Phiên trong bộ nhớ (không ghi file) — dùng để thử trước khi ghi state synthetic."""
    zt = _zaic_tools()
    cat = json.loads((SYN_ROOT / "synthetic_tools_catalog.json").read_text(encoding="utf-8")) \
        if (SYN_ROOT / "synthetic_tools_catalog.json").exists() else \
        json.loads((OFFICIAL / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))
    entry = {"_meta": build_meta(sample, qid), **copy.deepcopy(states)}
    return zt.Session(qid, entry, cat)


def replay(env, calls):
    """Chạy lần lượt; trả list kết quả, hoặc None nếu có call bị từ chối (ok=false)."""
    outs = []
    for c in calls:
        r = env.call(c["name"], c["arguments"])
        if not r.get("ok"):
            return None
        outs.append(r)
    return outs


def final_state_of(env, executed_calls=None, tmap=None, before: dict | None = None) -> dict:
    """State của các app bị thay đổi so với lúc mở phiên (Executor.replay)."""
    after = env.get_state()
    if before is None:
        before = getattr(env, "_lm_initial_state", None)
    if before is None:
        raise RuntimeError("cần before=state ban đầu (gọi mark_initial(env) ngay sau khi mở phiên)")
    return {app: st for app, st in after.items() if st != before.get(app)}


def mark_initial(env):
    env._lm_initial_state = env.get_state()
    return env


def check_roundtrip(samples, limit=None) -> dict:
    """Kiểm chứng: state lấy qua get_state() + _meta dựng lại => cùng hash, replay gold ra đúng
    tool_results và đúng label.final_state như phiên ToolEnv gốc."""
    from collections import Counter
    c = Counter()
    for s in samples[:limit] if limit else samples:
        env0 = train_env(s["id"])
        st = env0.get_state()
        h0 = env0.get_state_hash()
        env1 = session_from(s, st, s["id"])
        if env1.get_state_hash() != h0:
            c["hash_mismatch"] += 1
            continue
        before = env1.get_state()
        res = [env1.call(x["name"], x["arguments"]) for x in s["label"]["tool_calls"]]
        c["results_equal" if res == s["label"]["tool_results"] else "results_differ"] += 1
        fs = final_state_of(env1, before=before)
        c["final_state_equal" if fs == s["label"]["final_state"] else "final_state_differ"] += 1
    return dict(c)
