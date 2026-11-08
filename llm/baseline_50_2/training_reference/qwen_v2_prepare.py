"""Build the entire combined V2 training package before any full training starts."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import wave

HERE = Path(__file__).resolve().parent
LLM = HERE.parents[1]; LIFE = LLM.parent / "LifeMate"
sys.path.insert(0, str(LLM)); sys.path.insert(0, str(LLM / "finetune"))
from local_baseline import agent as baseline
from local_baseline.data import agent_view
from sft_data import make_examples
from policy import VERSION, TfidfExamples, enriched_body, safe_noise, selected, mask_app, normalize, words

EXTRA_SYSTEM = """\nThông tin number_time_hints chỉ là cách đọc số/giờ từ nguyên văn, không phải yêu cầu mới; không tự suy ngày/giờ khi thiếu căn cứ.
independent_training_example là minh họa từ một tác vụ khác: không thực thi nó, không sao chép ID, địa chỉ hay ngày giờ sang yêu cầu hiện tại.
Không tự thêm giá trị mặc định vào tham số tùy chọn. Chỉ truyền tham số tùy chọn khi nó cần để biểu đạt yêu cầu và có căn cứ từ input/kết quả tool.
Nếu thiếu thông tin định danh hoặc địa điểm, tìm bằng công cụ được phép trước; không dùng mô tả mơ hồ làm ID/địa chỉ thật.
Đọc kỹ phủ định, hủy, tự sửa ý và điều kiện. Lời trò chuyện/cảm ơn không tự động là yêu cầu gửi tin nhắn hay thao tác app.
"""
SYSTEM = baseline.SYSTEM + EXTRA_SYSTEM
NOISE_SOURCE = LIFE / "runs/20261104T153637Z_full_qwen35_2b_asr_8cd964/traces.jsonl"


def sha(path):
    with path.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf8")
    if path.exists():
        if path.read_bytes() != data: raise ValueError("Refuse to overwrite different prepared data: " + str(path))
    else:
        with path.open("xb") as f: f.write(data)


def json_new(path, value): new(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def demo_for(row, group):
    v = agent_view(row)
    mini = {"current_time": v['context']['current_time'], "timezone": v['context']['timezone']}
    history = [t for t in v['history'] if t['role']=='user'][-2:]
    request = v['latest_text']
    call = row['label']['tool_calls'][0] if row['label']['tool_calls'] else None
    visible = json.dumps({"context": mini, "history": history, "request": request}, ensure_ascii=False).lower()
    if call:
        for k, value in call['arguments'].items():
            if (k.endswith('_id') or k in ('id', 'phone_number', 'contact_id')) and isinstance(value, str) and value.lower() not in visible:
                return None
    result = {"source_id": row['id'], "group_id": group, "context": mini, "history": history,
              "request": request, "next_action": {"tool_call": call} if call else {"finish": True}}
    if len(json.dumps(result, ensure_ascii=False)) > 1800: return None
    return result


def noisy_row(row, mapping):
    result = copy.deepcopy(row); changed = 0; rejected = 0
    for target in [result] + [t for t in result.get('history', []) if t['role']=='user']:
        candidate = mapping.get(target['audio'])
        if candidate is None: continue
        clean = target['audio_text']
        if safe_noise(clean, candidate): target['audio_text'] = normalize(candidate); changed += 1
        elif words(clean) != words(candidate): rejected += 1
    return result, changed, rejected


def clips_for(ids, rows, base):
    clips = {}
    for i in ids:
        for turn in [rows[i]] + [t for t in rows[i].get('history', []) if t['role']=='user']:
            path = (base / turn['audio']).resolve()
            assert path.is_relative_to((base / 'audio').resolve())
            key = sha(path)
            with wave.open(str(path), 'rb') as w:
                assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
                duration = w.getnframes() / 16000
                assert duration <= 30
            item = {"audio": turn['audio'], "sha256": key, "text": normalize(turn['audio_text']), "duration_s": duration}
            if key in clips: assert clips[key]['text'] == item['text'], "Inconsistent annotation for identical audio"
            else: clips[key] = item
    return [clips[k] for k in sorted(clips)]


def build(output):
    cfg = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    split_path = LIFE / 'data/manifests/split-v1.json'
    split = json.loads(split_path.read_text(encoding='utf8'))
    base = LIFE / 'data/lifemate-v3.2.0/train+public_test/train'
    assert sha(base / 'train.jsonl') == split['train_jsonl_sha256']
    rows = {r['id']: r for r in map(json.loads, (base / 'train.jsonl').read_text(encoding='utf8').splitlines())}
    train_ids, val_ids = split['train_fit'], split['val_main']
    assert len(train_ids)==1236 and len(val_ids)==309 and not set(train_ids)&set(val_ids)
    assert not {split['group_ids'][i] for i in train_ids}&{split['group_ids'][i] for i in val_ids}
    demos = [d for i in train_ids if (d := demo_for(rows[i], split['group_ids'][i])) is not None]
    index = TfidfExamples(demos)
    noise = {}
    with NOISE_SOURCE.open(encoding='utf8') as f:
        for line in f:
            item = json.loads(line)
            if item['id'] not in set(train_ids) or not item.get('asr'): continue
            mapping = item['asr']['transcripts']
            assert hashlib.sha256(json.dumps(mapping, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == item['asr']['transcript_sha256']
            noise[item['id']] = mapping
    tools_json = json.loads((base / 'train_tools_catalog.json').read_text(encoding='utf8'))
    tools = {t['name']: t for app in tools_json['apps'] for t in app['tools']}
    stats = Counter(); manifest = {"version": cfg['version'], "policy": VERSION, "config": cfg,
        "train_sha256": sha(base/'train.jsonl'), "split_sha256": sha(split_path), "noise_source_sha256": sha(NOISE_SOURCE),
        "noise_source_asr": "frozen PhoWhisper-small, filter TRAIN-fit IDs before use; ignore all source LLM outputs",
        "retrieval_train_sources": len(demos), "partitions": {}}
    def messages(view):
        demo = index.retrieve(view, view['id'], split['group_ids'].get(view['id']))
        stats['fewshot_present'] += demo is not None
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(
            enriched_body(view, index, view['id'], split['group_ids'].get(view['id'])), ensure_ascii=False, separators=(',', ':'))}]
    for name, ids in [('train', train_ids), ('val', val_ids)]:
        examples = []
        for i in ids:
            row = copy.deepcopy(rows[i])
            if name=='train' and selected(i, 'asr_noise', .30):
                row, changed, rejected = noisy_row(row, noise.get(i, {}))
                stats['noise_selected_sources'] += 1; stats['noise_changed_sources'] += changed > 0
                stats['noise_changed_turns'] += changed; stats['noise_rejected_risky_turns'] += rejected
            if name=='train' and selected(i, 'app_mask', .10):
                row['audio_text'], changed = mask_app(row['audio_text'], row['context']['installed_apps'])
                stats['masked_app_sources'] += changed
            examples.extend(make_examples(row, tools, SYSTEM, messages, baseline.allowed_schemas))
        assert {e['source_id'] for e in examples} == set(ids)
        path = output / (name + '.jsonl')
        new(path, ''.join(json.dumps(e, ensure_ascii=False, separators=(',', ':'))+'\n' for e in examples))
        manifest['partitions'][name] = {"source_ids": ids, "sources": len(ids), "examples": len(examples),
            "sha256": sha(path), "calls": sum(e['kind']=='call' for e in examples), "finish": sum(e['kind']=='finish' for e in examples)}
        clips = clips_for(ids, rows, base)
        json_new(output / (name + '_asr_clips.json'), clips)
        manifest['partitions'][name].update(unique_audio=len(clips), audio_hours=sum(c['duration_s'] for c in clips)/3600)
    a={x['sha256'] for x in json.loads((output/'train_asr_clips.json').read_text(encoding='utf8'))}
    b={x['sha256'] for x in json.loads((output/'val_asr_clips.json').read_text(encoding='utf8'))}
    assert not a&b
    json_new(output / 'retrieval_index.json', demos)
    json_new(output / 'split.json', split)
    new(output / 'SYSTEM.txt', SYSTEM)
    new(output / 'config.json', (HERE/'config.json').read_text(encoding='utf8'))
    manifest['augmentation_counts'] = dict(stats)
    manifest['feature_source_sha256'] = {n:sha(HERE/n) for n in ('policy.py','prepare.py','config.json')}
    manifest['baseline_agent_sha256'] = sha(LLM/'local_baseline/agent.py')
    json_new(output / 'DATA_MANIFEST.json', manifest)
    print(json.dumps({"partitions": manifest['partitions'], "augmentation_counts": dict(stats), "retrieval_examples": len(demos)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.output.resolve())
