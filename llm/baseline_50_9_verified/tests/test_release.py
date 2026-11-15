import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from baseline50 import release
from baseline50.io import verify_assets

ROOT=Path(__file__).resolve().parents[1]
class ReleaseTests(unittest.TestCase):
    def test_frozen_core_and_package_hashes(self):
        release.verify()
    def test_medium_or_other_asset_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/"assets.manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"exact baseline50.9 Large"):
                verify_assets({"assets_dir":path})
    def test_manifest_metadata_is_large_identity(self):
        m=json.loads((ROOT/"assets.manifest.json").read_text())
        self.assertEqual(m["asr_parameters"],1543304960)
        self.assertEqual(m["merged_parameters"],5749056256)
        self.assertEqual(m["files"][m["qwen_file"]]["sha256"],
            "cb09b88189a7a1e001022b08eac0e7609c560cd3dee0bca37f9414a85f74a8b1")
    def test_payload_audit_rejects_secret_data_and_unknown_files(self):
        spec=importlib.util.spec_from_file_location("payload_audit",ROOT/"tools/audit_git_payload.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for name,text,reason in [("leak.jsonl","{}","Private/binary"),
                                ("unknown.md","unreviewed","Unreviewed"),
                                ("leak.txt","hf_"+"A"*30,"Possible binary/credential"),
                                ("pointer.txt","version https://git-lfs.github.com/spec/v1","Possible binary/credential")]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)
                approved={} if name=="unknown.md" else {name:hashlib.sha256(text.encode()).hexdigest()}
                (path/"CODE_MANIFEST.json").write_text(json.dumps({"files_sha256":approved}))
                (path/"PACKAGING_VERIFICATION.json").write_text("{}")
                (path/name).write_text(text)
                with self.assertRaisesRegex(ValueError,reason):module.audit(path)
    def test_import_and_verify_outside_original_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            code="import sys;sys.path.insert(0,sys.argv[1]);from baseline50 import run,release;release.verify();assert run.HERE==release.HERE"
            result=subprocess.run([sys.executable,"-B","-c",code,str(ROOT)],cwd=tmp,
                                  capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
