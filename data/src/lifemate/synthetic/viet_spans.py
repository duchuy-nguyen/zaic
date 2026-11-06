"""Tìm cụm SỐ TIỀN / GIỜ trong câu tiếng Việt và đổi ra giá trị, để t3 chỉ xóa ĐÚNG cụm
gắn với giá trị gold của tham số mục tiêu (bản cũ dùng re.sub xóa mọi con số trong câu).

  find_amount_spans(text) -> [(start, end, value_int)]
  find_time_spans(text)   -> [(start, end, (hour, minute))]
"""
from __future__ import annotations

import re
import unicodedata

DIGIT_WORD = {"không": 0, "linh": None, "lẻ": None, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4,
              "tư": 4, "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9}
SCALE = {"mười": 10, "mươi": 10, "trăm": 100}
BIG = {"nghìn": 1_000, "ngàn": 1_000, "k": 1_000, "triệu": 1_000_000, "tr": 1_000_000,
       "củ": 1_000_000, "tỷ": 1_000_000_000, "tỉ": 1_000_000_000}
NUM_WORDS = set(DIGIT_WORD) | set(SCALE) | {"rưỡi"}

TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+", re.UNICODE)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def _tokens(text: str):
    return [(m.start(), m.end(), m.group(0)) for m in TOKEN_RE.finditer(nfc(text))]


def _digits_value(tok: str) -> float | None:
    """'2' -> 2 ; '1,5' / '1.5' -> 1.5 ; '500.000' / '500,000' -> 500000"""
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", tok):
        return float(re.sub(r"[.,]", "", tok))
    if re.fullmatch(r"\d+[.,]\d{1,2}", tok):
        return float(tok.replace(",", "."))
    if tok.isdigit():
        return float(tok)
    return None


def _words_value(words: list[str]) -> int | None:
    """Đọc số < 1000 viết bằng chữ: 'một trăm hai mươi' -> 120, 'năm mươi' -> 50, 'hai' -> 2."""
    if not words:
        return None
    total, cur, seen = 0, 0, False
    i = 0
    while i < len(words):
        w = words[i]
        if w in ("linh", "lẻ"):
            i += 1
            continue
        if w in DIGIT_WORD and DIGIT_WORD[w] is not None:
            cur = cur * 1 + DIGIT_WORD[w] if cur == 0 else cur + DIGIT_WORD[w]
            seen = True
        elif w == "mười":
            cur += 10
            seen = True
        elif w == "mươi":
            cur = (cur or 1) * 10
            seen = True
        elif w == "trăm":
            total += (cur or 1) * 100
            cur = 0
            seen = True
        else:
            return None
        i += 1
    return total + cur if seen else None


def find_amount_spans(text: str):
    """Trả các cụm tiền: <số chữ/số> <nghìn|ngàn|k|triệu|tr|củ|tỷ> [rưỡi] [<số> (phần lẻ)] [đồng|đ]."""
    text = nfc(text)
    toks = _tokens(text)
    low = [t[2].lower() for t in toks]
    out = []
    i = 0
    while i < len(toks):
        # 1) đọc phần số
        j = i
        val = None
        if _digits_value(low[i]) is not None:
            val = _digits_value(low[i])
            j = i + 1
            m = re.fullmatch(r"(\d+(?:[.,]\d+)?)(k|tr)", low[i])
        else:
            m = re.fullmatch(r"(\d+(?:[.,]\d+)?)(k|tr)", low[i])
        if m:                                              # '50k', '2tr' dính liền
            val = float(m.group(1).replace(",", ".")) * BIG[m.group(2)]
            j = i + 1
            out.append((toks[i][0], toks[j - 1][1], int(round(val))))
            i = j
            continue
        if val is None:
            k = i
            while k < len(toks) and low[k] in NUM_WORDS and low[k] != "rưỡi":
                k += 1
            if k > i:
                val = _words_value(low[i:k])
                j = k
        if val is None or j >= len(toks) or low[j] not in BIG:
            i += 1
            continue
        # 2) đơn vị lớn
        unit = BIG[low[j]]
        amount = val * unit
        end = j + 1
        # 3) 'rưỡi' hoặc phần lẻ đơn vị kế ('1 triệu 2' -> 1.200.000, 'hai trăm rưỡi' đã xử lý ở trên)
        if end < len(toks) and low[end] == "rưỡi":
            amount += unit / 2
            end += 1
        elif end < len(toks) and unit >= 1_000_000:
            tail = None
            if _digits_value(low[end]) is not None and _digits_value(low[end]) < 1000:
                tail, nxt = _digits_value(low[end]), end + 1
            else:
                k = end
                while k < len(toks) and low[k] in NUM_WORDS and low[k] != "rưỡi":
                    k += 1
                if k > end:
                    tail, nxt = _words_value(low[end:k]), k
            if tail is not None:
                # '1 triệu 2' = 1.200.000 ; '1 triệu 250' = 1.250.000 ; nếu sau tail là 'nghìn' thì cộng nghìn
                if nxt < len(toks) and low[nxt] in ("nghìn", "ngàn", "k"):
                    amount += tail * 1000
                    end = nxt + 1
                else:
                    scale = unit // 10 if tail < 10 else (unit // 100 if tail < 100 else unit // 1000)
                    amount += tail * scale
                    end = nxt
        if end < len(toks) and low[end] in ("đồng", "đ", "vnđ", "vnd"):
            end += 1
        out.append((toks[i][0], toks[end - 1][1], int(round(amount))))
        i = end
    return out


PERIOD = {"sáng": "am", "trưa": "noon", "chiều": "pm", "tối": "pm", "đêm": "night"}
DAYW = {"mai", "nay", "mốt"}


def _hour_tokens(low, i):
    """Đọc giờ tại i: '9' | 'chín' | 'mười hai' ... ; trả (giờ, j) hoặc None."""
    v = _digits_value(low[i])
    if v is not None and float(v).is_integer() and 0 <= v <= 24:
        return int(v), i + 1
    if low[i] in NUM_WORDS and low[i] not in ("rưỡi", "không", "linh", "lẻ"):
        k = i
        while k < len(low) and low[k] in NUM_WORDS and low[k] != "rưỡi" and k - i < 3:
            k += 1
        val = _words_value(low[i:k])
        if val is not None and 0 <= val <= 24:
            return val, k
    return None


def find_time_spans(text: str):
    """Cụm giờ: [lúc|vào|tầm|khoảng] [sáng|trưa|chiều|tối] [mai|nay] <giờ> (giờ|h|rưỡi)
    [kém N | N phút | rưỡi] [sáng|trưa|chiều|tối|đêm] [mai|nay|ngày mai]
    -> (start, end, (h, m), core_start, core_end)
    core = phần GIỜ cần xóa ([lúc|vào|tầm|khoảng] + giờ + phút); buổi/ngày giữ lại
    (train: "nhắc anh uống liều Cardiol sáng mai" vẫn hỏi `time`)."""
    text = nfc(text)
    toks = _tokens(text)
    low = [t[2].lower() for t in toks]
    out = []
    i = 0
    while i < len(toks):
        start = i
        k = i
        while k < len(toks) and low[k] in ("lúc", "vào", "tầm", "khoảng", "ngày") | PERIOD.keys() | DAYW:
            k += 1
        if k >= len(toks):
            break
        ht = _hour_tokens(low, k)
        if not ht:
            i += 1
            continue
        core_s = k
        while core_s > start and low[core_s - 1] in ("lúc", "vào", "tầm", "khoảng"):
            core_s -= 1
        hour, j = ht
        minute = 0
        ok = False
        if j < len(toks) and low[j] in ("giờ", "h"):
            ok, j = True, j + 1
        if j < len(toks) and low[j] == "rưỡi":
            ok, minute, j = True, 30, j + 1
        elif ok and j < len(toks) and low[j] == "kém" and j + 1 < len(toks):
            mt = _hour_tokens(low, j + 1)
            if mt and mt[0] < 60:
                minute = 60 - mt[0]
                hour -= 1
                j = mt[1]
                if j < len(toks) and low[j] == "phút":
                    j += 1
        elif ok and j < len(toks):
            mt = _hour_tokens(low, j)
            if mt and mt[0] < 60 and (mt[1] >= len(toks) or low[mt[1]] in ("phút",) or mt[0] >= 10):
                minute = mt[0]
                j = mt[1]
                if j < len(toks) and low[j] == "phút":
                    j += 1
        if not ok:
            i += 1
            continue
        core_e = j
        period = None
        for w in low[start:k]:
            period = PERIOD.get(w, period)
        while j < len(toks) and (low[j] in PERIOD or low[j] in DAYW or low[j] == "ngày"):
            period = PERIOD.get(low[j], period)
            j += 1
        h24 = hour
        if period in ("pm", "night") and hour < 12:
            h24 = hour + 12
        if period == "noon" and hour < 11:
            h24 = hour + 12
        out.append((toks[start][0], toks[j - 1][1], (h24 % 24, minute),
                    toks[core_s][0], toks[core_e - 1][1]))
        i = j
    return out


SELF_CORRECTION = re.compile(r"à không|à nhầm|ấy nhầm|à mà thôi|à quên|không phải|à khoan|ờ khoan|à thôi", re.I)


def has_self_correction(text: str) -> bool:
    return bool(SELF_CORRECTION.search(nfc(text)))


def cleanup(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;])", r"\1", text)
    text = re.sub(r"([,;])\s*([,.!?;])", r"\2", text)
    text = re.sub(r"^\s*[,;]\s*", "", text)
    return text.strip()
