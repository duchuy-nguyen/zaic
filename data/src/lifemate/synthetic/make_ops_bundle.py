"""Bộ "train smoke" bàn giao cho OPS (HANDOFF_REVIEW_2026-11-05, mục "Bàn giao tiếp theo" 1):
  "vài sample TRAIN và environment tương ứng, train catalog/runtime riêng, root layout,
   archive/runtime/catalog hashes; manifest split và ID đã chốt."

Tạo thư mục <out>/ đúng root layout mà ToolEnvFactory(ToolEnv, root=<out>, split="train") của OPS yêu cầu:
  <out>/inputs/inputs_smoke.jsonl             input SẠCH (không label/state), transcript_source ghi rõ
  <out>/references_train_only/references_smoke.jsonl   gold tool_calls + gold state_hash — chỉ cho evaluator
  <out>/ids/expected_ids_smoke.jsonl (+ fit/val/public_test)   {"id": ...} theo PREFLIGHT.md
  <out>/train/train_tools_catalog.json        catalog train nguyên bản
  <out>/train/train_environment/<id>/state.json
  <out>/train/zaic_tools/                     runtime train nguyên bản (đưa <out>/train lên PYTHONPATH)
  <out>/manifest.json                         ids, loại mẫu, sha256 từng file, catalog/runtime/split/archive hashes
  <out>/provenance.data.json                  2 mục data + environment theo đúng format provenance của OPS
  <out>/verify_bundle.py                      tự kiểm (process mới, chỉ dùng runtime trong bundle)
  <out>/README.md

state.json KHÔNG copy file gốc: dựng từ ToolEnv("train").get_state() + _meta từ context (như pipeline synthetic),
rồi kiểm chứng state_hash trùng phiên ToolEnv gốc và replay gold ra đúng tool_results/final_state.

  python -m src.lifemate.synthetic.make_ops_bundle --out ops_train_smoke_v1 [--archive <đường_dẫn_zip_BTC>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from .common import MANIFEST, OFFICIAL, load_fit_ids, load_jsonl, tool_map
from .env_runner import _zaic_tools, build_meta, train_env
from ..eval.scorer import VERSION as EVAL_VERSION, behavior

VERSION = "ops_train_smoke_v1"
SEED = 20261109


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha(root: Path) -> str:
    """Hash cây thư mục (bỏ __pycache__/*.pyc): path\\0 len\\0 bytes, sắp theo path — cùng kiểu code_identity của OPS."""
    d = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    for p in files:
        data = p.read_bytes()
        d.update(p.relative_to(root).as_posix().encode() + b"\0")
        d.update(str(len(data)).encode() + b"\0" + data)
    return d.hexdigest()


def categorize(s, tmap):
    calls = s["label"]["tool_calls"]
    h = len(s.get("history", []))
    hb = "h0" if h == 0 else ("h1-4" if h <= 4 else "h5+")
    if not calls:
        return f"empty|{hb}"
    if calls[-1]["name"] == "sys.ask_missing_info":
        props = tmap[calls[-1]["arguments"]["tool_name"]]["parameters"].get("properties", {})
        kind = "entity" if any("x-ref" in props.get(p, {}) for p in calls[-1]["arguments"]["params"]) else "scalar"
        return f"ask_{kind}|{hb}"
    writes = [c for c in calls if tmap[c["name"]]["kind"] == "write"]
    reads = [c for c in calls if tmap[c["name"]]["kind"] == "read"]
    if not writes:
        return f"read_only|{hb}"
    return f"{'lookup_then_write' if reads else 'write_only'}|{hb}"


# 12 mẫu: phủ hành vi x độ dài history (thứ tự = ưu tiên)
PLAN = ["write_only|h0", "lookup_then_write|h0", "lookup_then_write|h1-4", "lookup_then_write|h5+",
        "read_only|h0", "ask_entity|h0", "ask_entity|h1-4", "ask_scalar|h0", "ask_scalar|h5+",
        "empty|h0", "empty|h1-4", "write_only|h5+"]


def choose(samples, tmap, n):
    rng = random.Random(SEED)
    by = {}
    for s in sorted(samples, key=lambda x: x["id"]):
        by.setdefault(categorize(s, tmap), []).append(s)
    for v in by.values():
        rng.shuffle(v)
    picked, apps = [], set()
    for cat in PLAN[:n]:
        pool = by.get(cat, [])
        # ưu tiên mẫu dùng app chưa có trong bundle -> đa dạng app
        pool = sorted(pool, key=lambda s: len({c["name"].split(".")[0] for c in s["label"]["tool_calls"]} & apps))
        if pool:
            s = pool[0]
            picked.append((cat, s))
            apps |= {c["name"].split(".")[0] for c in s["label"]["tool_calls"]}
    return picked


VERIFY = r'''"""Tự kiểm bundle — chạy trong PROCESS MỚI, chỉ dùng runtime của bundle:
    python verify_bundle.py            (từ thư mục bundle)
Kiểm: (1) hash catalog/runtime/file khớp manifest; (2) với từng mẫu: ToolEnv("train", id, root=bundle)
có state_hash ban đầu khớp; replay gold -> tool_results bằng label, state_hash cuối bằng expected.
Không đọc state.json trực tiếp."""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "train"))
from zaic_tools import ToolEnv  # runtime train của bundle


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tree_sha(root):
    d = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file() and "__pycache__" not in x.parts and x.suffix != ".pyc"):
        b = p.read_bytes()
        d.update(p.relative_to(root).as_posix().encode() + b"\0")
        d.update(str(len(b)).encode() + b"\0" + b)
    return d.hexdigest()


m = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
bad = []
if sha(ROOT / "train/train_tools_catalog.json") != m["catalog_sha256"]:
    bad.append("catalog_sha256")
if tree_sha(ROOT / "train/zaic_tools") != m["runtime_sha256"]:
    bad.append("runtime_sha256")
for rel, h in m["files"].items():
    if sha(ROOT / rel) != h:
        bad.append("file:" + rel)
def jl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


refs = {r["id"]: r for r in jl(ROOT / "references_train_only/references_smoke.jsonl")}
inputs = {r["id"]: r for r in jl(ROOT / "inputs/inputs_smoke.jsonl")}
exp = refs
if set(inputs) != set(refs) or any("label" in r or "tool_results" in r or "state_hash" in r for r in inputs.values()):
    bad.append("inputs_not_clean_or_ids_mismatch")
for sid, e in refs.items():
    env = ToolEnv("train", sid, root=ROOT)
    if env.get_state_hash() != e["initial_state_hash"]:
        bad.append(sid + ":initial_hash"); continue
    for c in e["tool_calls"]:
        if not env.call(c["name"], c["arguments"]).get("ok", True) and c["name"] != "sys.ask_missing_info":
            bad.append(sid + ":call_failed")
    if env.get_state_hash() != e["state_hash"]:
        bad.append(sid + ":gold_state_hash")
print(f"samples: {len(exp)} | errors: {len(bad)} {bad[:5]}")
print("BUNDLE OK" if not bad else "BUNDLE FAIL")
sys.exit(0 if not bad else 1)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=VERSION)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--archive", default=None, help="file zip dữ liệu BTC (để ghi archive_sha256); bỏ trống -> null")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "train" / "train_environment").mkdir(parents=True)
    for d in ("inputs", "references_train_only", "ids"):
        (out / d).mkdir()

    cat_path = OFFICIAL / "train" / "train_tools_catalog.json"
    catalog = json.loads(cat_path.read_text(encoding="utf-8"))
    tmap = tool_map(catalog)
    fit = load_fit_ids()
    train = [s for s in load_jsonl(OFFICIAL / "train" / "train.jsonl") if s["id"] in fit]
    picked = choose(train, tmap, a.n)

    shutil.copy2(cat_path, out / "train" / "train_tools_catalog.json")
    shutil.copytree(OFFICIAL / "train" / "zaic_tools", out / "train" / "zaic_tools",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    zt = _zaic_tools()
    rows, exp = [], []
    for cat, s in picked:
        env0 = train_env(s["id"])
        st, h0 = env0.get_state(), env0.get_state_hash()
        entry = {"_meta": build_meta(s, s["id"]), **st}
        d = out / "train" / "train_environment" / s["id"]
        d.mkdir()
        (d / "state.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        env1 = zt.ToolEnv("train", s["id"], root=out)                    # nạp lại từ bundle
        assert env1.get_state_hash() == h0, f"{s['id']}: hash state bundle khác phiên gốc"
        res = [env1.call(c["name"], c["arguments"]) for c in s["label"]["tool_calls"]]
        assert res == s["label"]["tool_results"], f"{s['id']}: replay gold khác label"
        rows.append(s)
        exp.append({"id": s["id"], "category": cat, "n_calls": len(s["label"]["tool_calls"]),
                    "apps": sorted({c["name"].split(".")[0] for c in s["label"]["tool_calls"]}),
                    "initial_state_hash": h0, "state_hash": env1.get_state_hash(),
                    "tool_calls": s["label"]["tool_calls"], "behavior": behavior(s["label"]["tool_calls"])})
    from ..eval.build_refs import clean_input

    def wj(p, rs):
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8", newline="\n")
    wj(out / "inputs" / "inputs_smoke.jsonl", [clean_input(r) for r in rows])
    wj(out / "references_train_only" / "references_smoke.jsonl",
       [{k: e[k] for k in ("id", "tool_calls", "state_hash", "behavior", "initial_state_hash")} for e in exp])
    wj(out / "ids" / "expected_ids_smoke.jsonl", [{"id": r["id"]} for r in rows])
    split_all = json.loads((MANIFEST / "split_v1.json").read_text(encoding="utf-8"))
    wj(out / "ids" / "expected_ids_fit.jsonl", [{"id": i} for i in sorted(split_all["fit"])])
    wj(out / "ids" / "expected_ids_val.jsonl", [{"id": i} for i in sorted(split_all["val"])])
    pub = None
    pt = OFFICIAL / "public_test"
    if (pt / "public_test.jsonl").exists():
        pids = [r["id"] for r in load_jsonl(pt / "public_test.jsonl")]
        wj(out / "ids" / "expected_ids_public_test.jsonl", [{"id": i} for i in pids])
        pcat = json.loads((pt / "public_test_tools_catalog.json").read_text(encoding="utf-8"))
        pub = {"n_ids": len(pids), "catalog_sha256": sha_file(pt / "public_test_tools_catalog.json"),
               "catalog_schema_version": pcat.get("schema_version"),
               "runtime_revision": f"lifemate-v3.2.0/public_test/zaic_tools (catalog schema {pcat.get('schema_version')})",
               "runtime_sha256": tree_sha(pt / "zaic_tools") if (pt / "zaic_tools").is_dir() else None,
               "n_tools": sum(len(a.get("tools", [])) for a in pcat["apps"]),
               "note": "runtime/env public_test lấy từ archive BTC (đúng archive_sha256); bundle không chứa public_test"}
    (out / "verify_bundle.py").write_text(VERIFY, encoding="utf-8", newline="\n")

    split_p = MANIFEST / "split_v1.json"
    files = {p.relative_to(out).as_posix(): sha_file(p) for p in sorted(out.rglob("*"))
             if p.is_file() and "zaic_tools" not in p.parts}
    archive_sha = sha_file(Path(a.archive)) if a.archive else None
    manifest = {
        "bundle_version": VERSION,
        "purpose": "OPS text smoke trên TRAIN thật (ToolEnvFactory split='train'); không phải submission/đánh giá chất lượng",
        "source": {"dataset": "lifemate-v3.2.0", "split": "train", "archive_file": Path(a.archive).name if a.archive else None,
                   "archive_sha256": archive_sha},
        "split_name": "split-v1", "split_manifest_sha256": sha_file(split_p),
        "train_fit_ids_sha256": sha_file(MANIFEST / "train_fit_provisional.json"),
        "sample_ids": [r["id"] for r in rows], "all_in_TRAIN_FIT": all(r["id"] in fit for r in rows),
        "categories": {e["id"]: e["category"] for e in exp},
        "catalog_sha256": sha_file(cat_path), "catalog_schema_version": catalog.get("schema_version"),
        "runtime_revision": f"lifemate-v3.2.0/train/zaic_tools (catalog schema {catalog.get('schema_version')})",
        "runtime_sha256": tree_sha(out / "train" / "zaic_tools"),
        "runtime_hash_rule": "sha256 over sorted files (excl. __pycache__, *.pyc): relpath\\0 len\\0 bytes",
        "state_json_origin": "ToolEnv('train').get_state() + _meta dựng từ context; hash trùng phiên gốc (đã kiểm)",
        "public_test": pub,
        "toolenv_api_confirmed": {
            "constructor": "ToolEnv(split, sample_id, root=<root>)  # root chứa <split>/<split>_tools_catalog.json, <split>/<split>_environment/<id>/state.json",
            "validate": "env.validate(tool, args) -> {'ok', 'error'}  (không chạy)",
            "call": "env.call(tool, args) -> dict có 'ok'; mọi call (cả lỗi) được log",
            "submission_fields": "env.get_tool_calls(), env.get_state_hash()",
            "train_only": "env.get_state() — chỉ debug/so label train; cấm trong đường inference",
            "source": "zaic_tools/README.md + executor.Session (đã chạy thử)"},
        "evaluator": {"version": EVAL_VERSION, "module": "data/src/lifemate/eval/scorer.py",
                      "references": "references_train_only/*.jsonl (id, tool_calls, state_hash) — không đưa vào inference",
                      "transcript_source": "inputs: gold_audio_text_train_only; public/private dùng ASR"},
        "files": files,
        "usage_rules": ["label/state chỉ dùng để so sánh trong test, không đưa vào input model",
                        "mỗi process chỉ load một zaic_tools: đưa <bundle>/train lên PYTHONPATH",
                        "dữ liệu BTC: không đưa lên git, phân phối qua Drive của đội"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    prov = {"data": {"archive_sha256": archive_sha, "split_name": "split-v1",
                     "split_manifest_sha256": manifest["split_manifest_sha256"],
                     "catalog_sha256": manifest["catalog_sha256"]},
            "environment": {"runtime_revision": manifest["runtime_revision"],
                            "runtime_sha256": manifest["runtime_sha256"]}}
    (out / "provenance.data.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    lines = "\n".join(f"| `{e['id']}` | {e['category'].replace('|', ' / ')} | {e['n_calls']} | {', '.join(e['apps']) or '—'} |" for e in exp)
    (out / "README.md").write_text(f"""# {VERSION} — bộ train smoke cho OPS

{len(rows)} mẫu **TRAIN_FIT** (split-v1) kèm environment, catalog và runtime train nguyên bản, đúng root layout của
`ToolEnvFactory(ToolEnv, root=<bundle>, split="train")`.

| id | loại | số call | app |
|---|---|---:|---|
{lines}

## Dùng
```bash
cd {VERSION}
python verify_bundle.py                       # phải in BUNDLE OK
PYTHONPATH=$PWD/train:<repo>/ops/src:<repo>/llm python ...   # một runtime / process
```
```python
from zaic_tools import ToolEnv                 # từ {VERSION}/train
from lifemate.adapters.toolenv import ToolEnvFactory
factory = ToolEnvFactory(ToolEnv, root=Path("{VERSION}"), split="train")
```
## Nội dung theo checklist "DATA bàn giao" (ops/docs/ops/CHECKLIST.md)
| Mục OPS | File |
|---|---|
| Archive/split/ID list/catalog SHA256 | `manifest.json`, `provenance.data.json`, `ids/expected_ids_{{smoke,fit,val,public_test}}.jsonl` |
| Source ToolEnv đúng split, runtime revision/hash, API | `train/zaic_tools`, `manifest.json` (`runtime_*`, `toolenv_api_confirmed`, `public_test`) |
| Input sạch label/state; history + transcript source | `inputs/inputs_smoke.jsonl` (`transcript_source` = gold_audio_text_train_only) |
| Evaluator version + references train-only tách inference | `references_train_only/references_smoke.jsonl`; scorer `data/src/lifemate/eval/scorer.py` ({EVAL_VERSION}) |

`references_train_only/references_smoke.jsonl`: gold `tool_calls`, `state_hash` sau khi replay gold, `initial_state_hash`
-> so với `session_output()` khi chạy đúng gold calls; chấm bằng
`python -m src.lifemate.eval.scorer --submission sub.jsonl --references references_smoke.jsonl --catalog train/train_tools_catalog.json`.

## API ToolEnv đã xác nhận (zaic_tools/README.md + executor.Session, đã chạy thử)
- `ToolEnv(split, sample_id, root=<root>)`; `env.validate(tool, args)` -> `{{ok, error}}` (không chạy);
  `env.call(tool, args)` -> dict có `ok` (mọi call, cả lỗi, đều được log);
  `env.get_tool_calls()`, `env.get_state_hash()` = hai trường của dòng nộp.
- `env.get_state()` chỉ dùng debug/so label TRAIN; cấm trong đường inference.

## Hash (điền provenance)
- split_name `split-v1`, split_manifest_sha256 `{manifest['split_manifest_sha256']}`
- catalog_sha256 `{manifest['catalog_sha256']}` (schema {catalog.get('schema_version')})
- runtime_sha256 `{manifest['runtime_sha256']}`
- archive_sha256 `{archive_sha}`
- public_test: `{json.dumps(pub, ensure_ascii=False) if pub else "không có official/public_test trên máy tạo bundle"}`
Chi tiết: `manifest.json`, `provenance.data.json`.

## Lưu ý
- `inputs/` không chứa label/state; `references_train_only/` chỉ cho evaluator. Public/private phải dùng transcript ASR.
- `state.json` dựng qua `get_state()` + `_meta` từ context (không copy file gốc); state_hash trùng phiên ToolEnv gốc.
- Đây là data của BTC: để trên Drive, không commit lên git.
""", encoding="utf-8", newline="\n")
    print(json.dumps({"out": str(out), "samples": len(rows), "categories": manifest["categories"],
                      "catalog_sha256": manifest["catalog_sha256"], "runtime_sha256": manifest["runtime_sha256"],
                      "split_manifest_sha256": manifest["split_manifest_sha256"], "archive_sha256": archive_sha},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
