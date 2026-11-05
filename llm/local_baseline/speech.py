"""Frozen local PhoWhisper-small provider. No gold transcripts, network or entity fixes."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import unicodedata
import wave

from .data import DATA_ROOT
from .setup_local import ROOT, sha256

MODEL_DIR = ROOT / ".runtime" / "models" / "PhoWhisper-small"


def normalize_transcript(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def audio_paths(row: dict) -> list[str]:
    # Deliberate audio-only access; never reads audio_text or label.
    return [row["audio"]] + [t["audio"] for t in row.get("history", []) if t["role"] == "user"]


def audio_view(row: dict, transcripts: dict[str, str]) -> dict:
    needed = audio_paths(row)
    if set(needed) - transcripts.keys():
        raise ValueError("ASR coverage missing; gold fallback is forbidden")
    history = []
    for turn in row.get("history", []):
        if turn["role"] == "user":
            history.append({"role": "user", "text": normalize_transcript(transcripts[turn["audio"]])})
        else:
            history.append({k: turn[k] for k in ("role", "text", "tool_calls", "tool_results") if k in turn})
    return {"id": row["id"], "latest_text": normalize_transcript(transcripts[row["audio"]]),
            "context": row["context"], "history": history}


class PhoWhisper:
    def __init__(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import numpy as np
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        if not torch.cuda.is_available():
            raise RuntimeError("Frozen ASR run requires CUDA; no silent CPU fallback")
        self.np, self.torch = np, torch
        random.seed(20261103)
        np.random.seed(20261103)
        torch.manual_seed(20261103)
        torch.cuda.manual_seed_all(20261103)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.processor = WhisperProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_DIR, local_files_only=True, torch_dtype=torch.float16, attn_implementation="sdpa"
        ).to("cuda").eval()
        self.model.generation_config.forced_decoder_ids = None
        self.manifest = json.loads((MODEL_DIR / "download_manifest.json").read_text(encoding="utf-8"))
        self.config = {"model": self.manifest["repository"], "revision": self.manifest["revision"],
                       "parameters": sum(p.numel() for p in self.model.parameters()),
                       "weights_manifest_sha256": sha256(MODEL_DIR / "download_manifest.json"),
                       "environment_lock_sha256": sha256(ROOT / ".runtime/asr-venv/requirements-frozen.txt"),
                       "device": "cuda", "dtype": "float16", "attention": "sdpa",
                       "language": "vi", "task": "transcribe", "num_beams": 1, "do_sample": False,
                       "max_new_tokens": 256, "batch_size": 4, "normalization": "NFC and whitespace only",
                       "history": "every user audio", "cached_transcripts": False, "seed": 20261103,
                       "torch": torch.__version__, "transformers": importlib.metadata.version("transformers")}

    def read_audio(self, relative: str):
        root = (DATA_ROOT / "train" / "audio").resolve()
        path = (DATA_ROOT / "train" / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("ASR input must remain within train/audio")
        with wave.open(str(path), "rb") as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000):
                raise ValueError("Expected original 16kHz mono PCM16; no silent resampling")
            frames = stream.readframes(stream.getnframes())
        values = self.np.frombuffer(frames, dtype="<i2").astype(self.np.float32) / 32768.0
        if len(values) > 30 * 16000:
            raise ValueError("Clip exceeds short-form ASR path; no truncation")
        return values

    def transcribe(self, paths: list[str], deadline: float) -> tuple[dict, dict]:
        started = time.monotonic()
        mapping, batches = {}, []
        # Do not reuse caches from earlier samples: this is a live per-sample timing run.
        for start in range(0, len(paths), 4):
            if time.monotonic() >= deadline:
                raise TimeoutError("ASR exhausted sample deadline")
            group = paths[start:start + 4]
            waveforms = [self.read_audio(path) for path in group]
            features = self.processor.feature_extractor(
                waveforms, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
            )
            inputs = features.input_features.to("cuda", dtype=self.torch.float16)
            attention = features.attention_mask.to("cuda")
            batch_started = time.monotonic()
            with self.torch.inference_mode():
                generated = self.model.generate(input_features=inputs, attention_mask=attention, language="vi",
                                                task="transcribe", do_sample=False, num_beams=1,
                                                max_new_tokens=256, return_timestamps=False,
                                                return_dict_in_generate=True)
                # Whisper's plain tensor path removes EOS during segment postprocessing.
                # The model-output path preserves raw sequences so truncation can be checked.
                ids = generated.sequences
            self.torch.cuda.synchronize()
            sequences = ids.cpu().tolist()
            eos = self.model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            if any(not eos_ids.intersection(sequence) for sequence in sequences):
                raise RuntimeError("ASR output hit token cap without EOS; no truncated transcript accepted")
            texts = self.processor.batch_decode(ids, skip_special_tokens=True)
            mapping.update({path: normalize_transcript(text) for path, text in zip(group, texts)})
            batches.append({"paths": group, "inference_s": time.monotonic() - batch_started,
                            "audio_s": [len(x) / 16000 for x in waveforms]})
            del inputs, attention, ids, features, generated
        digest = hashlib.sha256(json.dumps(mapping, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return mapping, {"elapsed_s": time.monotonic() - started, "clips": len(paths), "transcript_sha256": digest,
                         "batches": batches, "transcripts": mapping}
