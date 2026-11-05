# LifeMate Speech

PhoWhisper-small inference, audio audit and TRAIN-only ASR evaluation. This module belongs to the Speech lane; LLM prompting, tool execution and submission orchestration are separate components.

## Layout

- `src/lifemate_speech/`: audio reference loader, normalization, ASR, cache, metrics and weak entity parsers.
- `scripts/`: `audit_audio.py`, `audit_supplement.py`, `transcribe.py`, `eval_asr.py`.
- `tests/`: synthetic unit fixtures only; official data is not required for the test suite.
- [Architecture and progress](docs/architecture.md), [team project plan](../PROJECT_PLAN_2026-11-03.md).

## Environment and private work directory

Target Python 3.11/3.12; verified locally on Windows/Python 3.12.14. Linux/Kaggle, CUDA and Python 3.11 have not been directly tested. Use a separate virtual environment.

Run these PowerShell commands from the repository root. Set the work directory explicitly so moving the code does not move the existing model/cache/output artifacts:

```powershell
$env:LIFEMATE_SPEECH_WORK_DIR = 'D:\ZAIC\speech_work'
$env:HF_HOME = Join-Path $env:LIFEMATE_SPEECH_WORK_DIR 'hf_home'
$DataRoot = 'D:\ZAIC\train+public_test\train+public_test'
$SpeechCode = Join-Path (Get-Location) 'LifeMate\speech'
$SpeechPython = Join-Path $env:LIFEMATE_SPEECH_WORK_DIR '.venv\Scripts\python.exe'

# Only when creating a new environment:
python -m venv (Join-Path $env:LIFEMATE_SPEECH_WORK_DIR '.venv')
& $SpeechPython -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
& $SpeechPython -m pip install -r (Join-Path $SpeechCode 'requirements.txt')
& $SpeechPython -m pip check
```

If the environment already exists, skip creation/installation. The committed CPU lock captures the verified Windows/Python 3.12 packages; it is not a tested CUDA/Linux lock. To install the CPU lock, use the official CPU extra index. Do not install CPU torch into a Kaggle GPU environment.

`LIFEMATE_SPEECH_WORK_DIR` applies to all scripts, model cache, transcript cache and tests. Without it, the default is `LifeMate/speech_work` beside this module, ignored by Git. Paths are resolved and output/cache paths must stay inside the selected work root and outside DATA_ROOT. Audio paths from JSONL are relative to their split directory. DATA_ROOT is never modified.

## Verification

```powershell
& $SpeechPython -m unittest discover -s (Join-Path $SpeechCode 'tests') -v
& $SpeechPython (Join-Path $SpeechCode 'scripts\transcribe.py') --help
& $SpeechPython (Join-Path $SpeechCode 'scripts\eval_asr.py') --help
```

Tests cover normalization preservation, cache invalidation, sorted batches, structured EOS checks, missing/corrupt/long audio, resume/torn-tail recovery, history identity joins, full denominators, model hash mismatch and entity parser cases. Test fixtures are not model inference results.

## Audit and transcription smoke

```powershell
& $SpeechPython (Join-Path $SpeechCode 'scripts\audit_audio.py') --data-root $DataRoot --splits train public_test --out (Join-Path $env:LIFEMATE_SPEECH_WORK_DIR 'repo_audit_preview') --limit 5
& $SpeechPython (Join-Path $SpeechCode 'scripts\transcribe.py') --data-root $DataRoot --split train --out (Join-Path $env:LIFEMATE_SPEECH_WORK_DIR 'repo_smoke_new') --device cpu --batch-size 2 --limit 5 --local-files-only
```

Use a new output directory or `--resume`; the CLI refuses to overwrite an existing transcript run otherwise. `--local-files-only` requires the pinned snapshot already downloaded into the work root's `hf_models`. Omit it only when model download is intended. No official audio/transcripts are uploaded by the scripts. `--sample-ids train_0008 train_0063 train_0070 train_1594` selects targeted current/history smoke samples.

The ASR baseline remains `vinai/PhoWhisper-small`, revision `a86b604c346caf7148c37512eafe783a16420adb`. Model weight SHA256 is checked against revision LFS metadata: `9ec6f07c5bd321fc8f477cf778c43c3fd94b0e33a9f20283fc61cc6d22cbfded`.

Greedy settings: `num_beams=1`, `do_sample=False`, `language=vi`, `task=transcribe`, seed 0, eager attention and deterministic algorithms. CPU uses float32; CUDA defaults to float16 but is not yet validated. The repository integration only adds configurable work paths; it does not change decoding or normalization. Global code hashes differ from historical standalone runs because they include source paths/evaluation changes.

- Audio must be 16 kHz mono PCM 16-bit. Missing/corrupt/truncated files produce explicit errors.
- Audio over 30 s produces `AudioTooLong` and `audio_too_long`; no implicit truncation or chunking.
- One `normalize()` performs NFC and whitespace only. No case folding, number/date/name rewriting or gold fallback.
- Current and every user-history audio are included. `turn_index` indexes the entire history array, including assistant positions.
- Cache key includes audio byte SHA256, revision, decoding/config hash and normalization version, never sample ID.
- `--no-cache` disables reads AND writes. A cold cache run must populate entries before a warm run can demonstrate hits.
- Resume only retains successful rows matching audio/config/path; error rows retry, corrupt middle lines fail, a torn final line may be recovered.

Each run writes `transcripts.jsonl`, `errors.jsonl`, `run_manifest.json`. Manifest records actual argv/executable, code hash, weight hash verification, package versions, config/seed, hardware, batch sizes, timing and coverage.

## Timing and fixed audit diagnostics

`asr_latency_s` is equal allocation of measured batch wall time by actual batch size. Batch wall includes waveform read/features/transfer/generation/decode, excluding model load and CLI hash/header I/O. Those costs are included in total run wall. CUDA timing synchronizes the device.

Rows include `batch_id`, actual `batch_size` and `batch_wall_time_s`; do not sum the same batch wall repeatedly. Cache hits use zero active ASR latency/size/wall and retain source timing separately. Cross-sample batch allocations are not isolated sample wall times. CPU/cache measurements cannot establish GPU competition latency; live uncached end-to-end measurement is required.

Audit thresholds are diagnostic heuristics, not tuned on public_test: duration <0.25 s; silence absolute amplitude <=-40 dBFS; mostly silent ratio >=0.95; clipping amplitude >=0.999 with clip ratio >=0.001. Silence-only RMS is null. Invalid duration remains counted as a file but is excluded from duration statistics, with incomplete sample counts reported. Unreferenced WAVs have null sample/kind rather than guessed metadata. Percentiles use linear interpolation unless the supplement explicitly names another method.

## TRAIN-only evaluation

```powershell
& $SpeechPython (Join-Path $SpeechCode 'scripts\eval_asr.py') --data-root $DataRoot --transcripts (Join-Path $env:LIFEMATE_SPEECH_WORK_DIR 'smoke_final_limit5_cold\transcripts.jsonl') (Join-Path $env:LIFEMATE_SPEECH_WORK_DIR 'smoke_final_history_cold\transcripts.jsonl') --out (Join-Path $env:LIFEMATE_SPEECH_WORK_DIR 'repo_eval_smoke') --scope transcript_samples
```

Default `--scope all_train` includes all TRAIN references; it has NOT been run. Smoke uses `transcript_samples`, including all current/history references of selected samples, even missing/error hypotheses. Join is `(sample_id, kind, turn_index)`; duplicate/unknown keys and public_test hypotheses are rejected. Gold is evaluation-only, never supplied to ASR or cache.

WER retains case/punctuation and splits on whitespace (not linguistic Vietnamese word segmentation). CER includes normalized spaces. Missing/error hypotheses are empty strings and count as deletions. Empty references use max(1,n) denominators; empty groups report null. Scores include micro/macro for all/current/history.

`ENTITY_RULES_VERSION=weak-regex-v3`: money, time, date, phone, name/contact, address, english_term. Exact spans remain case-sensitive weak regex labels, not annotated truth. Reference-absent categories and false-positive-only hypotheses are reported separately.

- Money relaxed `vnd-fractions-guard-v2`: full fractional span, single trailing digit as tenths of preceding unit, single-scale `rưỡi` as half. Multi-digit tails without unit and compound-scale half remain unparseable. Fraction presence mismatch is incorrect, even if a base/truncated amount would agree. This conservative structural guard can reject an equivalent smaller-unit/unscaled spelling; review before full evaluation.
- Clock diagnostic `clock-kem-ruoi-v1`: subtract `kém`, add 30 minutes for `rưỡi`, parse spoken minutes. Missing AM/PM and day rollover are flagged, not inferred from context.
- Calendar dates exclude a following `lần`; `hôm hai chín` and `ngày mùng một tháng mười` are supported. Missing month/year/century remains unknown.
- Relative time `relative-phrase-mai-v2`: Mai after contact titles is excluded; bare Mai remains a weak label with ambiguity flags.

These diagnostics never modify transcript/normalization or WER/CER. Outputs include per-utterance metrics, aggregate metrics, up to 100 worst utterances, report, errors, 10 TRAIN reference examples/type, parser uncertainties and evaluation manifest. Reference examples are separate from the smoke score denominator.

## Current evidence and pending work

Historical standalone code: full audit 6,969 WAV; 25-audio/9-sample CPU smoke with cold/warm cache; WER micro 35.36%, CER micro 16.30% on that SMALL smoke set. Full TRAIN/public inference, full TRAIN evaluation, CUDA/RTX 4060 smoke and peak VRAM have not run. The source integration is checked with unit tests and smoke evaluation of existing private transcripts; it is not a new model benchmark.

Private audio, gold/hypotheses, evaluation samples, models, caches, environments and run outputs are intentionally absent from Git. Architecture contains aggregate results only; referenced local run artifacts require team-private storage. Full/GPU work remains gated by the user's confirmation.
