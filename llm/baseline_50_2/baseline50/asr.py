"""Frozen ASR decode and audio-view logic; explicit model/dataset paths."""
import hashlib,json,os,random,time,unicodedata,wave
from pathlib import Path

def normalize_transcript(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text).split())

def audio_paths(row: dict) -> list[str]:
    return [row['audio']] + [t['audio'] for t in row.get('history', []) if t['role'] == 'user']

def audio_view(row: dict, transcripts: dict[str, str]) -> dict:
    needed = audio_paths(row)
    if set(needed) - transcripts.keys():
        raise ValueError('ASR coverage missing; gold fallback is forbidden')
    history = []
    for turn in row.get('history', []):
        if turn['role'] == 'user':
            history.append({'role': 'user', 'text': normalize_transcript(transcripts[turn['audio']])})
        else:
            history.append({k: turn[k] for k in ('role', 'text', 'tool_calls', 'tool_results') if k in turn})
    return {'id': row['id'], 'latest_text': normalize_transcript(transcripts[row['audio']]), 'context': row['context'], 'history': history}

def torch_setup():
    import numpy as np
    import torch
    import transformers
    assert transformers.__version__ == '4.57.1' and torch.cuda.is_available()
    random.seed(20261103)
    np.random.seed(20261103)
    torch.manual_seed(20261103)
    torch.cuda.manual_seed_all(20261103)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return (np, torch)

class MediumASR:

    def __init__(self, folder, dataset_root, split, expected_vocab_sha256):
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.np, self.torch = torch_setup()
        self.dataset_root = Path(dataset_root).resolve()
        self.split = split
        self.train_warmup = False
        self.processor = WhisperProcessor.from_pretrained(folder, local_files_only=True)
        vocabulary = json.dumps(self.processor.tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False).encode()
        if hashlib.sha256(vocabulary).hexdigest() != expected_vocab_sha256:
            raise ValueError('Processor vocabulary differs from frozen release')
        self.model = WhisperForConditionalGeneration.from_pretrained(folder, local_files_only=True, use_safetensors=True, torch_dtype=self.torch.float16, attn_implementation='sdpa').to('cuda').eval()
        self.model.generation_config.forced_decoder_ids = None
        if sum((p.numel() for p in self.model.parameters())) != 763857920:
            raise ValueError('Expected merged PhoWhisper-medium')

    def read_audio(self, relative):
        split = 'train' if self.train_warmup else self.split
        root = (self.dataset_root / split / 'audio').resolve()
        path = (self.dataset_root / split / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Audio path escapes split/audio')
        with wave.open(str(path), 'rb') as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000):
                raise ValueError('Expected original 16kHz mono PCM16')
            frames = stream.readframes(stream.getnframes())
        values = self.np.frombuffer(frames, dtype='<i2').astype(self.np.float32) / 32768.0
        if len(values) > 30 * 16000:
            raise ValueError('Clip exceeds30s; no truncation')
        return values

    def warmup(self, train_id):
        with (self.dataset_root / 'train/train.jsonl').open(encoding='utf8') as stream:
            path = next((r['audio'] for r in map(json.loads, stream) if r['id'] == train_id))
        self.train_warmup = True
        try:
            self.transcribe([path], time.monotonic() + 120)
        finally:
            self.train_warmup = False

    def transcribe(self, paths: list[str], deadline: float) -> tuple[dict, dict]:
        started = time.monotonic()
        mapping, batches = ({}, [])
        for start in range(0, len(paths), 4):
            if time.monotonic() >= deadline:
                raise TimeoutError('ASR exhausted sample deadline')
            group = paths[start:start + 4]
            waveforms = [self.read_audio(path) for path in group]
            features = self.processor.feature_extractor(waveforms, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
            inputs = features.input_features.to('cuda', dtype=self.torch.float16)
            attention = features.attention_mask.to('cuda')
            batch_started = time.monotonic()
            with self.torch.inference_mode():
                generated = self.model.generate(input_features=inputs, attention_mask=attention, language='vi', task='transcribe', do_sample=False, num_beams=1, max_new_tokens=256, return_timestamps=False, return_dict_in_generate=True)
                ids = generated.sequences
            self.torch.cuda.synchronize()
            sequences = ids.cpu().tolist()
            eos = self.model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            if any((not eos_ids.intersection(sequence) for sequence in sequences)):
                raise RuntimeError('ASR output hit token cap without EOS; no truncated transcript accepted')
            texts = self.processor.batch_decode(ids, skip_special_tokens=True)
            mapping.update({path: normalize_transcript(text) for path, text in zip(group, texts)})
            batches.append({'paths': group, 'inference_s': time.monotonic() - batch_started, 'audio_s': [len(x) / 16000 for x in waveforms]})
            del inputs, attention, ids, features, generated
        digest = hashlib.sha256(json.dumps(mapping, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return (mapping, {'elapsed_s': time.monotonic() - started, 'clips': len(paths), 'transcript_sha256': digest, 'batches': batches, 'transcripts': mapping})
