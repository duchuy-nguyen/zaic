"""Frozen PhoWhisper inference, sorted mini-batches, no gold input.

Files >30 s return AudioTooLong explicitly; no implicit processor truncation.
latency_s = measured batch wall time / actual batch size, not isolated file latency.
Batch wall time includes waveform read, features, transfer, generation and decode;
model initialization/download is separate. CUDA is synchronized around timing.
"""
from __future__ import annotations
import copy
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import random
import time
import wave

from .io_utils import audio_sha256, ensure_work_path, speech_work_dir, wav_info

MODEL_ID = 'vinai/PhoWhisper-small'
REVISION = 'a86b604c346caf7148c37512eafe783a16420adb'
SEED = 0
MODEL_WEIGHT_SHA256 = '9ec6f07c5bd321fc8f477cf778c43c3fd94b0e33a9f20283fc61cc6d22cbfded'


def model_weight_provenance(cache_dir):
    path = Path(cache_dir) / 'models--vinai--PhoWhisper-small' / 'snapshots' / REVISION / 'pytorch_model.bin'
    actual = audio_sha256(path) if path.is_file() else None
    result = dict(path=str(path.resolve()), filename='pytorch_model.bin', sha256=actual,
                  expected_lfs_sha256=MODEL_WEIGHT_SHA256, revision=REVISION,
                  expected_hash_source=f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true',
                  verified=actual == MODEL_WEIGHT_SHA256)
    if actual is not None and actual != MODEL_WEIGHT_SHA256:
        raise ValueError('Downloaded model weight SHA256 does not match pinned revision LFS hash')
    return result


def package_versions():
    result = {}
    for name in ('torch', 'transformers', 'numpy', 'huggingface-hub', 'tokenizers', 'safetensors'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def decoding_config(device='cpu', dtype=None, batch_size=2):
    dtype = dtype or ('float32' if device == 'cpu' else 'float16')
    return dict(model_id=MODEL_ID, revision=REVISION, num_beams=1, do_sample=False,
                model_weight_sha256=MODEL_WEIGHT_SHA256,
                language='vi', task='transcribe', temperature=0.0, forced_decoder_ids=None,
                return_timestamps=False, max_length=448, seed=SEED,
                return_dict_in_generate=True,
                dtype=dtype, device=device, batch_size=batch_size,
                attention_backend='eager', deterministic_algorithms=True,
                audio_policy='pcm16-mono-16khz-reject-over30-v1',
                batch_sort='duration-ascending-stable-v1', libraries=package_versions())


class PhoWhisperASR:
    def __init__(self, model_id=MODEL_ID, revision=REVISION, device='cpu', dtype=None, batch_size=2,
                 model_cache_dir=None, local_files_only=False):
        if model_id != MODEL_ID or revision != REVISION:
            raise ValueError('This baseline requires the frozen PhoWhisper model and revision')
        if device not in ('cpu', 'cuda') or batch_size < 1:
            raise ValueError('Invalid device or batch_size')
        self.config = decoding_config(device, dtype, batch_size)
        if self.config['dtype'] not in ('float32', 'float16') or (device == 'cpu' and self.config['dtype'] != 'float32'):
            raise ValueError('CPU requires float32; CUDA accepts float16/float32')
        self.device, self.batch_size = device, batch_size
        default_cache = speech_work_dir() / 'hf_models'
        self.model_cache_dir = str(ensure_work_path(Path(model_cache_dir) if model_cache_dir else default_cache))
        self.local_files_only = local_files_only
        self.model = self.processor = None
        self.model_load_s = 0.0
        self.load_error = None
        self.effective_generation_config = None
        self.hardware = {}

    def _load(self):
        if self.model is not None:
            return
        if self.load_error:
            raise RuntimeError(self.load_error)
        started = time.perf_counter()
        try:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            os.environ.setdefault('HF_HOME', str(speech_work_dir() / 'hf_home'))
            import numpy as np
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            torch.use_deterministic_algorithms(True)
            if self.device == 'cuda':
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA requested but unavailable')
                torch.cuda.manual_seed_all(SEED)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
            kwargs = dict(revision=REVISION, cache_dir=self.model_cache_dir, local_files_only=self.local_files_only)
            model_weight_provenance(self.model_cache_dir)
            processor = WhisperProcessor.from_pretrained(MODEL_ID, **kwargs)
            model = WhisperForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype=getattr(torch, self.config['dtype']), attn_implementation='eager', **kwargs)
            if not model_weight_provenance(self.model_cache_dir)['verified']:
                raise ValueError('Pinned model weight file not available for verification')
            model.to(self.device).eval()
            gen = copy.deepcopy(model.generation_config)
            for name in ('num_beams', 'do_sample', 'language', 'task', 'temperature', 'forced_decoder_ids', 'return_timestamps', 'max_length', 'return_dict_in_generate'):
                setattr(gen, name, self.config[name])
            model.generation_config = gen
            self.processor, self.model = processor, model
            self.effective_generation_config = gen.to_dict()
            self.hardware = dict(torch_threads=torch.get_num_threads(), cuda_runtime=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name(0) if self.device == 'cuda' else None,
                                 logical_parameter_count=sum(p.numel() for p in model.parameters()))
        except Exception as exc:
            self.load_error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            self.model_load_s += time.perf_counter() - started

    def transcribe_batch(self, paths: list[Path]) -> list[dict]:
        results, valid = [None] * len(paths), []
        for i, path in enumerate(paths):
            try:
                info = wav_info(path)
                valid.append((info['audio_duration_s'], i, path))
            except Exception as exc:
                results[i] = dict(text_raw=None, latency_s=None, audio_duration_s=None, batch_size=0,
                                  batch_wall_time_s=None, error=f'{type(exc).__name__}: {exc}')
        valid.sort(key=lambda item: item[0])
        for offset in range(0, len(valid), self.batch_size):
            group = valid[offset:offset+self.batch_size]
            started, wall = None, None
            try:
                self._load()
                import numpy as np
                import torch
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                started = time.perf_counter()
                audio = []
                for _, _, path in group:
                    with wave.open(str(path), 'rb') as stream:
                        expected = stream.getnframes()
                        data = stream.readframes(expected)
                    if len(data) != expected * 2:
                        raise ValueError(f'Truncated PCM data: {path}')
                    audio.append(np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768)
                features = self.processor(audio, sampling_rate=16000, return_tensors='pt',
                                          padding='max_length', truncation=False, return_attention_mask=True)
                features = features.to(self.device)
                features['input_features'] = features['input_features'].to(getattr(torch, self.config['dtype']))
                with torch.inference_mode():
                    # Plain Whisper tensor output strips EOS in this Transformers revision.
                    # Structured output preserves decoder prefix/EOS for truncation checks.
                    output = self.model.generate(**features).sequences
                texts = self.processor.batch_decode(output, skip_special_tokens=True)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                wall = time.perf_counter() - started
                eos = self.model.generation_config.eos_token_id
                for (duration, i, _), text, tokens in zip(group, texts, output.tolist(), strict=True):
                    error = None if eos in tokens else 'GenerationTruncated: no EOS before max_length; output rejected'
                    results[i] = dict(text_raw=text if error is None else None, latency_s=wall/len(group),
                                      audio_duration_s=duration, batch_size=len(group), batch_wall_time_s=wall, error=error)
            except Exception as exc:
                if started is not None:
                    wall = time.perf_counter() - started
                for duration, i, _ in group:
                    results[i] = dict(text_raw=None, latency_s=wall/len(group) if wall is not None else None,
                                      audio_duration_s=duration, batch_size=len(group), batch_wall_time_s=wall,
                                      error=f'{type(exc).__name__}: {exc}')
        return results
