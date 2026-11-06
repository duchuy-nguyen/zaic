from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
import sys

for _s in (sys.stdout, sys.stderr):           # Windows cmd / log file: in tiếng Việt không lỗi mã hoá
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Chạy mọi lệnh từ thư mục LifeMate/ (giống bản cũ)
OFFICIAL = Path("official/lifemate-v3.2.0")
SYN_ROOT = Path("synthetic")
MANIFEST = Path("manifests")
RUNS = Path("runs")

PHONE_RE = re.compile(r"^0\d{9,10}$")


def norm_text(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "").lower()).strip()


def tokens(s):
    return re.findall(r"[0-9a-zà-ỹ]+", norm_text(s))


def token_f1(a, b):
    ta, tb = Counter(tokens(a)), Counter(tokens(b))
    inter = sum((ta & tb).values())
    if not inter:
        return 0.0
    p, r = inter / sum(ta.values()), inter / sum(tb.values())
    return 2 * p * r / (p + r)


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_catalog(split="train"):
    return json.loads((OFFICIAL / split / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))


def tool_map(cat):
    return {t["name"]: t for a in cat["apps"] for t in a.get("tools", [])}


def app_group(cat):
    return {a["id"]: a.get("group") for a in cat["apps"]}


def gold_ns(s):
    return {c["name"].split(".")[0] for c in s["label"]["tool_calls"] if not c["name"].startswith("sys.")}


def xref_params(sc):
    return [k for k, v in sc.get("parameters", {}).get("properties", {}).items() if "x-ref" in v]


def load_fit_ids():
    return set(json.loads((MANIFEST / "train_fit_provisional.json").read_text(encoding="utf-8")))


def load_val_ids():
    m = json.loads((MANIFEST / "split_v1.json").read_text(encoding="utf-8"))
    return set(m["val"])


# ---- KHÔNG còn load_state() đọc thẳng train_environment/<id>/state.json ----
# Luật: "must not read, copy or parse any file under <split>_environment/".
# State của train lấy qua API được phép cho train: ToolEnv(...).get_state()  -> xem env_runner.train_state()


def collect_names(o, acc=None, keys=("name", "title", "device_name", "account_name", "display_name")):
    """Bản cũ chỉ lấy key 'name' -> bỏ sót ~500 ứng viên t4 (diag_t4). Giờ lấy thêm các key tên khác."""
    acc = [] if acc is None else acc
    if isinstance(o, dict):
        for k in keys:
            if isinstance(o.get(k), str) and o[k]:
                acc.append(o[k])
        for v in o.values():
            collect_names(v, acc, keys)
    elif isinstance(o, list):
        for v in o:
            collect_names(v, acc, keys)
    return acc


def audio_refs(samples):
    for s in samples:
        yield s["audio"]
        for t in s.get("history", []):
            if t.get("role") == "user" and t.get("audio"):
                yield t["audio"]
