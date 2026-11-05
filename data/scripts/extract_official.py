import zipfile, hashlib
from pathlib import Path
ZIP = Path("train+public_test.zip")
OUT = Path("official/lifemate-v3.2.0")
EXPECTED = {
 "train+public_test/train/train.jsonl":"24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8",
 "train+public_test/train/train_tools_catalog.json":"a994984bb580d7e0d60166641858901d428549682ba3614dbf9269037a33cce0",
 "train+public_test/public_test/public_test.jsonl":"6274f4ff32ab03c33350029a0672a67884209b7d97f66e80181e446dd955f604",
 "train+public_test/public_test/public_test_tools_catalog.json":"c8f46c278a7495d36905ac28fc5b85b67aad22bc2ac0b29ba9ac6455ad9daf5d",
}
def sha(b):
    h=hashlib.sha256(); h.update(b); return h.hexdigest()
with zipfile.ZipFile(ZIP) as z:
    for info in z.infolist():
        n=info.filename
        if "__MACOSX" in n or "/._" in n or n.endswith(".DS_Store") or info.is_dir(): continue
        rel=n.split("train+public_test/",1)[-1]
        tgt=OUT/rel; tgt.parent.mkdir(parents=True,exist_ok=True)
        data=z.read(n)
        if n in EXPECTED: assert sha(data)==EXPECTED[n], f"HASH SAI: {n}"
        tgt.write_bytes(data)
print("hash OK ->", OUT)