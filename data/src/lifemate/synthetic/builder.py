import json
import random
import shutil

from .common import OFFICIAL, SYN_ROOT


ID_TAG = "v21"   # v2.1: ID mới, không trùng ID của bản v2 / release filtered-311 (synth_t2_00014 ...)


class SyntheticBuilder:
    """Quản lý split 'synthetic' (cùng layout với split official để ToolEnv chạy được)."""

    def __init__(self, tech, fit_ids, seed=20261103):
        self.tech, self.fit_ids, self.rng = tech, fit_ids, random.Random(seed)
        self.records, self.n = [], 0
        SYN_ROOT.mkdir(parents=True, exist_ok=True)
        (SYN_ROOT / "synthetic_environment").mkdir(exist_ok=True)
        if not (SYN_ROOT / "zaic_tools").exists():                      # code, không phải data environment
            shutil.copytree(OFFICIAL / "train" / "zaic_tools", SYN_ROOT / "zaic_tools",
                            ignore=shutil.ignore_patterns("__pycache__"))
        if not (SYN_ROOT / "synthetic_tools_catalog.json").exists():
            shutil.copy2(OFFICIAL / "train" / "train_tools_catalog.json", SYN_ROOT / "synthetic_tools_catalog.json")

    def assert_source(self, sid):
        assert sid in self.fit_ids, f"LEAK RISK: {sid} ngoai TRAIN_FIT"

    def new_id(self):
        self.n += 1
        return f"synth_{ID_TAG}_{self.tech}_{self.n:05d}"

    def write_state(self, syn_id, sample, states):
        """Ghi synthetic_environment/<syn_id>/state.json = _meta dựng từ context + states (đã/không đột biến)."""
        from .env_runner import build_meta
        d = SYN_ROOT / "synthetic_environment" / syn_id
        d.mkdir(parents=True, exist_ok=True)
        entry = {"_meta": build_meta(sample, syn_id), **states}
        (d / "state.json").write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8", newline="\n")

    def write_state_from_train(self, syn_id, sample):
        """State gốc (không đột biến) của mẫu train, lấy qua ToolEnv.get_state()."""
        from .env_runner import train_state
        self.write_state(syn_id, sample, train_state(sample["id"]))

    def drop_state(self, syn_id):
        shutil.rmtree(SYN_ROOT / "synthetic_environment" / syn_id, ignore_errors=True)
        self.n -= 1

    def add(self, rec):
        self.records.append(rec)

    def finalize(self):
        self.records.sort(key=lambda r: r["id"])
        out = SYN_ROOT / f"phase1_{self.tech}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.records), encoding="utf-8", newline="\n")
        return {"file": str(out), "n": len(self.records)}
