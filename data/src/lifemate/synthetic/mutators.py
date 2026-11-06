"""Đột biến state cho T2: thêm một thực thể TRÙNG TÊN với thực thể mà lệnh ghi cuối của gold trỏ tới.

Sửa so với bản cũ:
  * ID bản sao theo đúng kiểu train (c_05 -> c_11, chat_03 -> chat_06, light_bedroom -> light_bedroom3),
    KHÔNG còn hậu tố "_syn" (train có 0 ID kiểu đó; mô hình sẽ học "thấy _syn thì hỏi").
  * Bản sao giữ nguyên các thuộc tính được nhắc trong câu, đổi thuộc tính KHÔNG được nhắc
    (vd relation: train có "Minh/đồng nghiệp" vs "Minh/khách hàng"); bản cũ chép y hệt 229/259 mẫu.
  * Số điện thoại mới dạng 09xxxxxxxxx (train 497/497 hợp lệ); bản cũ xáo chữ số -> 22/30 không hợp lệ.
"""
from __future__ import annotations

import copy
import json
import re

from .common import norm_text

NAME_KEYS = ("name", "title")
PHONE_KEYS = ("phone", "phone_number", "msisdn")
SWAPPABLE = {"relation"}          # thuộc tính được phép đổi nếu câu không nhắc tới
PHONE_RE = re.compile(r"^0\d{9,10}$")


def find_entity(o, eid, parent=None):
    """Tìm dict có id == eid; trả (entity, list_cha) — chỉ nhận thực thể nằm trong một list."""
    if isinstance(o, dict):
        if o.get("id") == eid and isinstance(parent, list):
            return o, parent
        for v in o.values():
            r = find_entity(v, eid, o)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_entity(v, eid, o)
            if r:
                return r
    return None


def all_ids(o, acc=None):
    acc = set() if acc is None else acc
    if isinstance(o, dict):
        if isinstance(o.get("id"), str):
            acc.add(o["id"])
        for v in o.values():
            all_ids(v, acc)
    elif isinstance(o, list):
        for v in o:
            all_ids(v, acc)
    return acc


def new_id_like(eid: str, siblings: list[str], taken: set[str], rng) -> str:
    m = re.fullmatch(r"(.*?)(\d+)", eid)
    if not m:                                        # light_bedroom -> light_bedroom2, 3, ...
        k = 2
        while f"{eid}{k}" in taken:
            k += 1
        return f"{eid}{k}"
    prefix, num = m.group(1), m.group(2)
    width = len(num)
    nums = []
    for s in siblings:
        mm = re.fullmatch(re.escape(prefix) + r"(\d+)", s or "")
        if mm:
            nums.append(int(mm.group(1)))
    sequential = bool(nums) and max(nums) <= len(nums) + 3     # c_01..c_07 -> nối tiếp; course_910, 586 -> ngẫu nhiên
    if sequential:
        k = max(nums) + 1
        while f"{prefix}{k:0{width}d}" in taken:
            k += 1
        return f"{prefix}{k:0{width}d}"
    lo, hi = 10 ** (width - 1), 10 ** width - 1     # id ngẫu nhiên kiểu course_910 -> cùng số chữ số
    for _ in range(1000):
        cand = f"{prefix}{rng.randint(lo, hi)}"
        if cand not in taken:
            return cand
    raise RuntimeError("không sinh được id mới")


def new_phone_like(old: str, taken: set[str], rng) -> str:
    n = len(old) if PHONE_RE.match(str(old)) else 11
    for _ in range(1000):
        cand = "09" + "".join(str(rng.randint(0, 9)) for _ in range(n - 2))
        if cand not in taken:
            return cand
    raise RuntimeError("không sinh được số điện thoại mới")


def mentioned(value, text_norm: str) -> bool:
    v = norm_text(str(value))
    return bool(v) and v in text_norm


def duplicate_for_ambiguity(states: dict, ref_id: str, audio_text: str, relation_pool: list[str], rng):
    """Đột biến `states` tại chỗ. Trả dict mô tả, hoặc None nếu không làm được một cách tự nhiên."""
    found = find_entity(states, ref_id)
    if not found:
        return None
    ent, parent = found
    name_key = next((k for k in NAME_KEYS if isinstance(ent.get(k), str) and ent[k].strip()), None)
    if not name_key:
        return None
    text_n = norm_text(audio_text)
    if not mentioned(ent[name_key], text_n):          # thực thể phải được gọi đúng tên trong câu
        return None
    same_name = [e for e in parent if isinstance(e, dict) and norm_text(str(e.get(name_key))) == norm_text(ent[name_key])]
    if len(same_name) > 1:                            # đã mơ hồ sẵn -> gold gốc lẽ ra đã là ask
        return None
    taken_ids = all_ids(states)
    dump = json.dumps(states, ensure_ascii=False)
    dup = copy.deepcopy(ent)
    dup["id"] = new_id_like(ref_id, [e.get("id") for e in parent if isinstance(e, dict) and isinstance(e.get("id"), str)],
                            taken_ids, rng)
    changed = {"id": [ref_id, dup["id"]]}
    for k in PHONE_KEYS:
        if dup.get(k):
            if mentioned(dup[k], text_n):
                return None                           # câu nói số điện thoại -> không mơ hồ
            taken_ph = set(re.findall(r"0\d{9,10}", dump))
            dup[k] = new_phone_like(str(dup[k]), taken_ph, rng)
            changed[k] = [ent[k], dup[k]]
    for k in SWAPPABLE:
        if isinstance(dup.get(k), str) and not mentioned(dup[k], text_n):
            pool = [r for r in relation_pool if r != dup[k] and not mentioned(r, text_n)]
            if pool:
                dup[k] = rng.choice(pool)
                changed[k] = [ent[k], dup[k]]
    parent.append(dup)
    return {"name": ent[name_key], "new_id": dup["id"], "changed": changed}
