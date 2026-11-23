"""Shared V2 input policy for both SFT and inference. No tool state or labels here."""
from collections import Counter
import hashlib
import math
import re
import unicodedata

VERSION = "v2-input-itn-tfidf-safe-noise-v1"
DIGITS = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
          "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9}
SCALES = {"nghìn": 1000, "ngàn": 1000, "triệu": 1000000, "tỷ": 1000000000}
NUMBER_WORDS = set(DIGITS) | set(SCALES) | {"mười", "mươi", "trăm", "linh", "lẻ"}
PROTECTED = NUMBER_WORDS | {"không", "đừng", "khỏi", "hủy", "huỷ", "thôi", "nếu", "thì", "trừ", "âm",
                          "trước", "sau", "sáng", "trưa", "chiều", "tối", "đêm", "mai", "nay", "qua", "giờ", "phút", "rưỡi", "kém"}
NUMBER_PATTERN = "(?:" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r"|\d+)"
BRANDS = {"chat_ott": ["zalo"], "media_player": ["spotify", "youtube"], "video_meeting": ["zoom", "google meet"]}


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def words(text):
    return re.findall(r"[^\W_]+", normalize(text).lower())


def vietnamese_number(text):
    ts = words(text)
    if len(ts) == 1 and ts[0].isdigit(): return int(ts[0])
    if not ts or any(t not in NUMBER_WORDS for t in ts): raise ValueError("Unsupported number phrase")
    total, section, digit, last_scale = 0, 0, None, 10**12
    for t in ts:
        if t in DIGITS:
            if digit is not None: raise ValueError("Ambiguous consecutive digits; use phone parser")
            digit = DIGITS[t]
        elif t == "trăm": section += (digit if digit is not None else 1) * 100; digit = None
        elif t in ("mười", "mươi"): section += (digit if digit is not None else 1) * 10; digit = None
        elif t in ("linh", "lẻ"): continue
        elif t in SCALES:
            scale = SCALES[t]
            if scale >= last_scale: raise ValueError("Unsupported nested scales")
            chunk = section + (digit or 0)
            if chunk == 0: raise ValueError("Missing scale amount")
            total += chunk * scale; section, digit, last_scale = 0, None, scale
    return total + section + (digit or 0)


def number_hints(text):
    """Hints retain the original phrase; never replace negation, dates, IDs or names."""
    original = normalize(text); lower = original.lower(); hints = []
    time_pattern = re.compile(r"(?<!\w)(?P<h>" + NUMBER_PATTERN + r"(?:\s+" + NUMBER_PATTERN + r")*)\s*(?P<sep>giờ|rưỡi)(?P<rest>(?:\s+[^\W_]+){0,7})", re.UNICODE)
    for m in time_pattern.finditer(lower):
        try:
            hour = vietnamese_number(m['h'])
            if not 0 <= hour <= 23: continue
            rest = words(m['rest'])
            if "kém" in rest[:3]: continue
            minute = 30 if m['sep'] == "rưỡi" or (rest and rest[0] == "rưỡi") else 0
            if rest and rest[0] in NUMBER_WORDS:
                mt = []
                for token in rest:
                    if token not in NUMBER_WORDS: break
                    mt.append(token)
                if mt: minute = vietnamese_number(" ".join(mt))
            if not 0 <= minute <= 59: continue
            period = next((t for t in rest if t in ("sáng", "trưa", "chiều", "tối", "đêm")), None)
            if period in ("chiều", "tối") and 1 <= hour < 12: hour += 12
            if period == "trưa" and hour == 1: hour = 13
            if period == "sáng" and hour == 12: hour = 0
            hints.append({"kind": "clock", "source": m.group(0), "hour": hour, "minute": minute,
                          "period": period, "meridiem_resolved": period is not None or hour > 12})
        except ValueError: continue
    money = re.compile(r"(?<!\w)" + NUMBER_PATTERN + r"(?:\s+" + NUMBER_PATTERN + r")*(?:\s+(?:đồng|vnđ|vnd))?(?!\w)", re.UNICODE)
    for m in money.finditer(lower):
        phrase = m.group(0); ts = words(phrase)
        if not (set(ts) & set(SCALES) or ts[-1] in ("đồng", "vnđ", "vnd")): continue
        if re.search(r"(?:âm|trừ)\s*$", lower[max(0, m.start()-8):m.start()]) or re.match(r"\s+rưỡi", lower[m.end():]): continue
        if ts[-1] in ("đồng", "vnđ", "vnd"): ts = ts[:-1]
        try: amount = vietnamese_number(" ".join(ts))
        except ValueError: continue
        hints.append({"kind": "money", "source": phrase, "value": str(amount), "currency": "VND"})
    return hints


def protected_signature(text):
    ts = words(text)
    critical = [t for t in ts if t in PROTECTED or any(c.isdigit() for c in t)]
    proper = [w.lower() for j, w in enumerate(re.findall(r"[^\W_]+", normalize(text)))
              if j > 0 and w[0].isupper()]
    return critical, proper


def distance(a, b):
    row = list(range(len(b) + 1))
    for j, x in enumerate(a, 1):
        new = [j]
        for k, y in enumerate(b, 1): new.append(min(row[k]+1, new[-1]+1, row[k-1]+(x != y)))
        row = new
    return row[-1]


def safe_noise(clean, noisy):
    a, b = words(clean), words(noisy)
    if a == b or not a: return False
    ac, names = protected_signature(clean); bc, _ = protected_signature(noisy)
    return ac == bc and all(name in b for name in names) and distance(a, b) / len(a) <= .30


def selected(sample_id, tag, fraction):
    return int(hashlib.sha256((tag + ":" + sample_id).encode()).hexdigest()[:12], 16) / 16**12 < fraction


def mask_app(text, installed):
    if len(installed) != 1: return text, False
    aliases = BRANDS.get(next(iter(installed)), [])
    for alias in aliases:
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        if re.search(pattern, text, re.I):
            # Preserve all context/schema; only remove a brand when one installed app makes routing unambiguous.
            return normalize(re.sub(pattern, "", text, flags=re.I)), True
    return text, False


class TfidfExamples:
    def __init__(self, examples):
        self.examples = examples
        counts = [Counter(words(e['request'])) for e in examples]
        df = Counter(t for c in counts for t in c)
        self.idf = {t: math.log((1+len(counts))/(1+n)) + 1 for t, n in df.items()}
        self.vectors = [self.vector(c) for c in counts]

    def vector(self, counts):
        v = {t: (1 + math.log(n)) * self.idf[t] for t, n in counts.items() if t in self.idf}
        norm = math.sqrt(sum(x*x for x in v.values())) or 1
        return {t: x/norm for t, x in v.items()}

    def retrieve(self, view, source_id=None, group_id=None):
        query = self.vector(Counter(words(view['latest_text'])))
        allowed = {t for app in view['context']['installed_apps'].values() for t in app['tools']} | {'sys.ask_missing_info'}
        best, best_score = None, .20
        for example, v in zip(self.examples, self.vectors):
            if example['source_id'] == source_id or (group_id is not None and example['group_id'] == group_id): continue
            call = example['next_action'].get('tool_call')
            if call and call['name'] not in allowed: continue
            score = sum(value * v.get(t, 0) for t, value in query.items())
            if score > best_score: best, best_score = example, score
        if best is None: return None
        return {k: best[k] for k in ('context', 'history', 'request', 'next_action')}


def enriched_body(view, retriever, source_id=None, group_id=None):
    body = {"context": view['context'], "history_already_executed": view['history'], "CURRENT_USER_REQUEST": view['latest_text']}
    hints = {"current": number_hints(view['latest_text']), "history": [
        {"turn": j, "hints": number_hints(t['text'])} for j, t in enumerate(view['history']) if t['role']=='user' and number_hints(t['text'])]}
    if hints['current'] or hints['history']: body['number_time_hints_not_new_requests'] = hints
    example = retriever.retrieve(view, source_id, group_id)
    if example is not None: body['independent_training_example_do_not_execute_or_copy_ids'] = example
    return body
