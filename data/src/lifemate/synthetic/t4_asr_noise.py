"""T4 — giả lập ASR nghe nhầm TÊN THỰC THỂ; nhãn gold giữ nguyên (mô hình phải đối chiếu mờ với kết quả read).

Sửa so với bản cũ (chỉ ra 3 mẫu):
  * Bản cũ chèn nhiễu vào BẤT KỲ token nào (phần lớn là động từ/hư từ: 'gọi'->'vợ', 'đặt'->'đã'),
    rồi so MỘT token nhiễu với tên đầy đủ bằng token-F1 -> gần như không mẫu nào qua; collect_names chỉ lấy 'name'.
  * Bản mới: chỉ làm nhiễu token NẰM TRONG tên của thực thể mà lệnh ghi trỏ tới (x-ref), tên đó phải được
    nhắc trong câu; tên lấy từ tool_results gold (name/title/...). Câu nhiễu chỉ đổi đúng 1 token, giữ nguyên
    phần còn lại (bản cũ hạ chữ thường cả câu).
  * Groundability: cụm tên nhiễu, sau khi bỏ dấu, phải giống tên đúng NHẤT và duy nhất trong danh sách read
    (SequenceMatcher >= 0.75) — giống cách zaic_tools tìm kiếm bỏ dấu (_fold).
  * Bỏ mẫu nếu tên gốc nằm trong một tham số chấm TUYỆT ĐỐI (vd contacts.search.query) — nhãn sẽ bắt mô hình
    đoán ngược tên gốc; cho phép trong tham số fuzzy dài >= 4 token (token-F1 vẫn > 0.6).
  * Bảng nhầm lẫn dựng lại từ FIT mới (t4b_confusion -> confusion_fit_v1.json).
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata

from .common import collect_names, norm_text, tokens, tool_map, xref_params
from .env_runner import replay, session_from, train_state

SIM_MIN = 0.75


TONES = {"̀", "́", "̃", "̉", "̣"}


def same_word_spelling(a: str, b: str) -> bool:
    """'hoà' vs 'hòa', 'thuỷ' vs 'thủy': cùng chữ, chỉ khác vị trí dấu thanh -> không phải lỗi ASR."""
    def key(s):
        d = unicodedata.normalize("NFD", s.lower())
        return "".join(c for c in d if c not in TONES), sorted(c for c in d if c in TONES)
    return key(a) == key(b)


def fold(s: str) -> str:
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def _entity_name(results, eid):
    for r in results:
        stack = [r]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                if o.get("id") == eid:
                    n = collect_names({k: v for k, v in o.items() if not isinstance(v, (dict, list))})
                    return n[0] if n else None
                stack.extend(o.values())
            elif isinstance(o, list):
                stack.extend(o)
    return None


def _candidates_names(results):
    names = set()
    for r in results:
        stack = [r]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                if "id" in o:
                    n = collect_names({k: v for k, v in o.items() if not isinstance(v, (dict, list))})
                    if n:
                        names.add(n[0])
                stack.extend(o.values())
            elif isinstance(o, list):
                stack.extend(o)
    return names


def _name_in_exact_arg(calls, name, tmap):
    nn = norm_text(name)
    for c in calls:
        props = tmap[c["name"]]["parameters"].get("properties", {})
        for k, v in c["arguments"].items():
            if isinstance(v, str) and nn and nn in norm_text(v):
                fuzzy = props.get(k, {}).get("x-match") == "fuzzy"
                if not fuzzy or len(tokens(v)) < 4:
                    return True
    return False


def build_t4(b, samples, catalog, conf, target=250):
    tmap = tool_map(catalog)
    cands = [s for s in samples if s["label"]["tool_calls"]
             and not any(c["name"].startswith("sys.") for c in s["label"]["tool_calls"])]
    b.rng.shuffle(cands)
    made, why = 0, {}

    def skip(k):
        why[k] = why.get(k, 0) + 1

    for s in cands:
        if made >= target:
            break
        calls, results = s["label"]["tool_calls"], s["label"]["tool_results"]
        targets = []
        for i, c in enumerate(calls):
            if tmap[c["name"]]["kind"] != "write":
                continue
            for p in xref_params(tmap[c["name"]]):
                eid = c["arguments"].get(p)
                if isinstance(eid, str):
                    name = _entity_name(results[:i], eid)
                    if name and norm_text(name) in norm_text(s["audio_text"]):
                        targets.append((name, eid))
        if not targets:
            skip("no_named_xref_target"); continue
        name, eid = b.rng.choice(targets)
        if _name_in_exact_arg(calls, name, tmap):
            skip("name_in_exact_arg"); continue
        text = s["audio_text"]
        m = re.search(re.escape(name), text, flags=re.I)
        if not m:
            skip("mention_not_found_casefold"); continue
        name_toks = [(mm.start() + m.start(), mm.end() + m.start(), mm.group(0))
                     for mm in re.finditer(r"[^\W_]+", text[m.start():m.end()])]
        options = [(st, en, tok, alt, n) for st, en, tok in name_toks
                   for alt, n in conf.get(tok.lower(), {}).items()
                   if alt != tok.lower() and not same_word_spelling(alt, tok)]
        if not options:
            skip("no_confusion_for_name_tokens"); continue
        weights = [o[4] for o in options]
        st, en, tok, alt, _ = b.rng.choices(options, weights=weights, k=1)[0]
        alt_cased = alt[:1].upper() + alt[1:] if tok[:1].isupper() else alt
        noisy_text = text[:st] + alt_cased + text[en:]
        noisy_mention = noisy_text[m.start():m.end() - (en - st) + len(alt_cased)]
        pool = _candidates_names(results)
        scored = sorted(((difflib.SequenceMatcher(None, fold(noisy_mention), fold(n)).ratio(), n) for n in pool),
                        reverse=True)
        if not scored or scored[0][1] != name or scored[0][0] < SIM_MIN or \
                (len(scored) > 1 and scored[1][0] >= scored[0][0]):
            skip("not_uniquely_groundable"); continue
        states = train_state(s["id"])
        env = session_from(s, states, "probe")
        full = replay(env, calls)
        if full is None or full != results:
            skip("replay_mismatch"); continue
        b.assert_source(s["id"])
        sid = b.new_id()
        b.write_state(sid, s, states)
        b.add({"id": sid, "source_id": s["id"], "technique": "t4_asr_noise", "audio": s["audio"],
               "audio_source": f"train:{s['id']}", "transcript_mode": "asr_noisy_sim", "audio_text": noisy_text,
               "context": s["context"], "history": s.get("history", []),
               "label": {"tool_calls": calls, "tool_results": full, "final_state": s["label"]["final_state"]},
               "provenance": {"original_text": text, "entity_id": eid, "entity_name": name,
                              "noise_ops": [{"gold": tok, "noisy": alt_cased}], "noisy_mention": noisy_mention,
                              "ground_sim": round(scored[0][0], 3), "conf_source": "phowhisper-small/train_fit_v1"}})
        made += 1
    print("t4 lý do loại:", why)
    return made
