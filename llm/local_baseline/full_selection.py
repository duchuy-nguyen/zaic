"""Select exactly the full labeled train set or the original 1236 remainder."""
import json
from .data import MANIFEST, prepare, read_rows
from .setup_local import ROOT, sha256

PATH = ROOT / "data/manifests/full-train-1545-v1.json"


def select(partition, split):
    actual = {r["id"] for r in read_rows()}
    all_ids = set(split["train_fit"]) | set(split["val_main"])
    if len(actual) != 1545 or all_ids != actual or len(split["train_fit"]) != 1236:
        raise ValueError("Full labeled train set differs from the frozen 1545/1236 selection")
    if set(split["train_fit"]) & set(split["val_main"]):
        raise ValueError("Old and remaining samples overlap")
    if partition == "train-extra":
        return sorted(split["train_fit"])
    if partition == "all-train":
        return sorted(actual)
    raise ValueError("Unsupported full-evaluation partition")


def create():
    split = prepare()
    record = {"version": "full-train-1545-v1", "split_sha256": sha256(MANIFEST),
              "all_train": select("all-train", split), "remaining_1236": select("train-extra", split),
              "original_309": split["val_main"], "smoke_included": split["smoke_train_fit"],
              "purpose": "Zero-shot evaluation on all original train examples; may train on these later; not a blind test"}
    if PATH.exists():
        if json.loads(PATH.read_text(encoding="utf-8")) != record:
            raise ValueError("Frozen full selection differs")
    else:
        with PATH.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
    return record


def expected_full_source(source):
    replacements = [
        ('choices=["val-main", "smoke", "expanded"]', 'choices=["val-main", "smoke", "expanded", "train-extra", "all-train"]'),
        ('    elif args.ids_manifest is not None:\n', '''    elif args.partition in ("train-extra", "all-train"):
        if args.limit is not None:
            parser.error("Full evaluation cannot be truncated with --limit")
        from .full_selection import select, create
        create()
        ids = select(args.partition, split)
    elif args.ids_manifest is not None:
'''),
        ('    code = {name:', '    if args.partition in ("train-extra", "all-train"):\n        critical.append("full_selection.py")\n    code = {name:'),
        ('    run_dir = args.resume.resolve()', '''    if args.partition in ("train-extra", "all-train"):
        from .full_selection import PATH as FULL_SELECTION_PATH
        common["full_selection_manifest_sha256"] = sha256(FULL_SELECTION_PATH)
    run_dir = args.resume.resolve()'''),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("Unexpected original runner source")
        source = source.replace(before, after)
    return source


if __name__ == "__main__":
    record = create()
    print("FULL_SELECTION", PATH, len(record["all_train"]), len(record["remaining_1236"]))
