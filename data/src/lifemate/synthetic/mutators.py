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

# v2.1 (review OPS 07/11): "Mẹ", "Bố", "Sếp Dũng", "Chị Ly", "Evergreen Foods" bị nhân đôi -> không tự nhiên.
# Train chỉ nhân đôi TÊN RIÊNG trơn: Minh 28, Lan 27, Hương 8, Bình, Trang (+1 "Bác sĩ An").
HONORIFIC = {"mẹ", "má", "mạ", "bố", "ba", "cha", "tía", "u", "ông", "bà", "bác", "chú", "cô", "dì", "cậu",
             "mợ", "thím", "dượng", "anh", "chị", "em", "con", "cháu", "sếp", "thầy", "vợ", "chồng", "thằng",
             "út", "ngoại", "nội", "bé", "nhóc", "lễ", "phòng", "ban", "nhóm", "team", "tổ", "gia"}
ORG_WORDS = {"foods", "food", "company", "co", "ltd", "jsc", "shop", "store", "group", "tòa", "toà", "công",
             "cửa", "hàng", "quán", "tiệm", "spa", "clinic", "bệnh", "viện", "trường", "lớp", "nhà"}
# Thực thể không phải người: bản sao phải khác ở một thuộc tính MÔ TẢ (lấy giá trị thật từ thực thể cùng list),
# thuộc tính đó không được nhắc trong câu. Không có thì bỏ (vd máy lọc không khí cùng phòng, bài hát đã nêu ca sĩ).
DESCRIPTIVE_KEYS = ("provider", "author", "artist", "teacher", "organizer", "host", "branch")
# (bỏ location/room/floor: đổi vị trí ngẫu nhiên ra tổ hợp vô lý, vd "cổng" ở "tầng 2")
# Từ chỉ vai trò đứng ngay trước tên: nếu có, đổi relation của bản sao sẽ khiến câu tự phân biệt được
# ("nhắn partner Kevin Park", "gọi thầy Peter Hansen") -> nhãn ask sai -> bỏ.
ROLE_BEFORE_NAME = {"partner", "đối tác", "thầy", "cô giáo", "giáo sư", "sếp", "bác sĩ", "khách", "khách hàng",
                    "đồng nghiệp", "hàng xóm", "vợ", "chồng", "con", "bạn thân", "thằng", "boss", "trưởng phòng"}
PERSON_LISTS = {"contacts"}                 # bản sao người: khác id + phone + relation (như train)
# v2.1: train có chat_01/chat_02 'Minh' giống hệt trừ id, nhưng DATA quyết định KHÔNG sinh bản sao chỉ-khác-ID
# (OPS 07/11 loại 18 mẫu kiểu này). Muốn bật lại: IDENTICAL_OK_LISTS = {"chats"}.
IDENTICAL_OK_LISTS: set = set()
PRONOUNS = {"anh", "chị", "em", "ông", "bà", "cô", "chú", "bác", "cậu", "dì", "con", "cháu", "thằng", "nó", "mình"}
RELATION_STOP = PRONOUNS | {"của", "và", "với", "cho", "bên", "ở"}


def is_person_name(name: str) -> bool:
    toks = name.split()
    if not toks or len(toks) > 4:
        return False
    low = [t.lower().strip(".,") for t in toks]
    if low[0] in HONORIFIC or any(t in ORG_WORDS for t in low):
        return False
    return all(t[:1].isupper() for t in toks)


def _list_key(states, parent):
    """Tên key chứa list cha (vd 'contacts', 'chats', 'books')."""
    stack = [states]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                if v is parent:
                    return k
                stack.append(v)
        elif isinstance(o, list):
            stack.extend(o)
    return None


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
    lkey = _list_key(states, parent)
    person_like = lkey in PERSON_LISTS or lkey in IDENTICAL_OK_LISTS or lkey == "chats"
    if lkey in IDENTICAL_OK_LISTS or (lkey not in PERSON_LISTS and lkey in {"chats"}):
        if lkey not in IDENTICAL_OK_LISTS:
            return {"skip": "id_only_clone_disabled"}
    if person_like and not is_person_name(ent[name_key]):
        return {"skip": "not_plain_person_name"}
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
    if lkey in PERSON_LISTS:
        before = text_n.split(norm_text(ent[name_key]))[0].split()
        for n in (2, 1):
            if len(before) >= n and " ".join(before[-n:]) in ROLE_BEFORE_NAME:
                return {"skip": "role_word_before_name"}
        rel = dup.get("relation")
        if not isinstance(rel, str) or mentioned(rel, text_n):
            return {"skip": "relation_missing_or_mentioned"}
        # OPS 07/11 (synth_t2_00047 "Toàn trực ca" vs relation "quản lý trực ca"): từ quanh tên trùng với
        # relation thì câu tự phân biệt được -> bỏ. Cửa sổ: 2 từ trước, 3 từ sau tên; bỏ qua đại từ xưng hô.
        parts = text_n.split(norm_text(ent[name_key]))
        near = set(parts[0].split()[-2:]) | (set(parts[1].split()[:3]) if len(parts) > 1 else set())
        near -= RELATION_STOP
        if near & (set(norm_text(rel).split()) - RELATION_STOP):
            return {"skip": "qualifier_near_name_matches_relation"}
        pool = [r for r in relation_pool if r != rel and not mentioned(r, text_n)
                and not (near & (set(norm_text(r).split()) - RELATION_STOP))]
        if not pool:
            return {"skip": "no_relation_alternative"}
        dup["relation"] = rng.choice(pool)
        changed["relation"] = [rel, dup["relation"]]
    elif lkey not in IDENTICAL_OK_LISTS:
        opts = []
        for k in DESCRIPTIVE_KEYS:
            v = ent.get(k)
            if not isinstance(v, str) or mentioned(v, text_n):
                continue
            alts = sorted({e.get(k) for e in parent if isinstance(e, dict) and isinstance(e.get(k), str)
                           and e.get(k) != v and not mentioned(e.get(k), text_n)})
            if alts:
                opts.append((k, alts))
        if not opts:
            return {"skip": "no_descriptive_difference"}
        k, alts = opts[0]
        dup[k] = rng.choice(alts)
        changed[k] = [ent[k], dup[k]]
    parent.append(dup)
    return {"name": ent[name_key], "new_id": dup["id"], "changed": changed, "list": lkey}
