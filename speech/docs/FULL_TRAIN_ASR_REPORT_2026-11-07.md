# Full TRAIN Speech — GPU inference and evaluation

Run date: 2026-11-07. Documentation updated: 2026-11-08.

Inference only: no training/fine-tuning, no public_test inference, no LLM/ToolEnv run.

## Coverage and provenance

- 5036/5036 successful audio; 0 errors; 0 missing rows.
- 1545 current + 3491 user-history audio, 1545 samples; identity/path/audio SHA256 validated on every file.
- PhoWhisper-small revision a86b604c346caf7148c37512eafe783a16420adb; model weight SHA256 9ec6f07c5bd321fc8f477cf778c43c3fd94b0e33a9f20283fc61cc6d22cbfded.
- Source/config matches accepted GPU smoke; CUDA float16, batch size 2, greedy, language vi, task transcribe, seed 0.
- --no-cache --resume: no cache reads/writes, from_cache=0; errors remain in denominator.
- Code SHA256 a6b89b7c3f552989e69522bddb3cfcf29481ab33677f5cd3390d8c70b98c36b3.
- Source commit: `a8bd8a6a7f3ed601f23be2e695928ed39d520df7`.
- Environment: Windows, Python 3.12.14, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 4060 Laptop GPU. See [CUDA dependency lock](../requirements-cuda-win-py312.lock.txt).
- Normalization: `nfc-whitespace-v1`; entity rules: `weak-regex-v3`; money relaxed: `vnd-fractions-guard-v2`; clock: `clock-kem-ruoi-v1`; relative time: `relative-phrase-mai-v2`.
- 19 fixture tests passed before the full run. Coverage/hash validation passed after the run; no missing or empty successful transcript.

- Overlap with accepted GPU smoke: 0/22 raw transcript/error differences. Global duration-sorted batch membership differs from the small smoke; no causal attribution or automatic transcript correction.

## WER / CER

| Scope | N audio | N reference words | WER micro | WER macro | CER micro | CER macro |
|---|---:|---:|---:|---:|---:|---:|
| all | 5036 | 72063 | 46.76% | 48.98% | 23.23% | 24.25% |
| current | 1545 | 34686 | 45.08% | 45.84% | 22.63% | 22.99% |
| history | 3491 | 37377 | 48.31% | 50.37% | 23.81% | 24.82% |

Case/punctuation retained; NFC + whitespace only, CER includes spaces. WER uses whitespace units, not Vietnamese word segmentation. Missing/error hypotheses count as empty strings.

## Weak entity errors — all audio

| Entity | Reference-present denominator | Exact | Incorrect | Unparseable | Error/denominator |
|---|---:|---:|---:|---:|---:|
| money | 188 | 72 | 116 | 0 | 116/188 (61.70%) |
| time | 719 | 262 | 457 | 0 | 457/719 (63.56%) |
| date | 268 | 137 | 131 | 0 | 131/268 (48.88%) |
| phone | 43 | 9 | 34 | 0 | 34/43 (79.07%) |
| name/contact | 717 | 150 | 567 | 0 | 567/717 (79.08%) |
| address | 75 | 6 | 69 | 0 | 69/75 (92.00%) |
| english_term | 86 | 51 | 35 | 0 | 35/86 (40.70%) |

Strict extracted-span multiset match; weak regex labels, not human annotations. Casing/address boundaries can inflate diagnostic errors. False-positive-only counts are separate in metrics.json.

## Separate diagnostics

Money relaxed: {"sentences_with_entity": 188, "exact_match_count": 113, "exact_match_rate": 0.601063829787234, "incorrect_count": 58, "unparseable_count": 17, "absent_count": 4848, "false_positive_only_count": 4, "fraction_presence_mismatch_count": 10}
Clock relaxed: {"sentences_with_entity": 719, "exact_match_count": 320, "exact_match_rate": 0.44506258692628653, "incorrect_count": 374, "unparseable_count": 25, "absent_count": 4317, "false_positive_only_count": 14}
Relative-time phrases: {"sentences_with_entity": 1358, "exact_match_count": 879, "exact_match_rate": 0.6472754050073638, "incorrect_count": 479, "unparseable_count": 0, "absent_count": 3678, "false_positive_only_count": 84}

| Relative phrase | Reference-present sentences | Missing/wrong phrase count |
|---|---:|---:|
| ngay bây giờ | 4 | 1 |
| ngày kia | 1 | 1 |
| ngày mai | 16 | 5 |
| sáng mai | 270 | 107 |
| trưa mai | 1 | 0 |
| chiều mai | 38 | 16 |
| tối mai | 46 | 21 |
| sáng mốt | 1 | 0 |
| sáng nay | 116 | 30 |
| trưa nay | 28 | 11 |
| chiều nay | 115 | 32 |
| tối nay | 122 | 54 |
| đêm nay | 1 | 0 |
| hôm nay | 192 | 24 |
| hôm qua | 41 | 9 |
| tối qua | 18 | 10 |
| tuần sau | 42 | 11 |
| tuần tới | 2 | 0 |
| tháng sau | 3 | 0 |
| tháng tới | 3 | 1 |
| năm sau | 1 | 0 |
| năm tới | 1 | 1 |
| bây giờ | 19 | 0 |
| mai | 347 | 144 |
| mốt | 12 | 1 |

## Timing / memory

- Total run wall: 2936.6 s (48.9 min); batch ASR wall sum: 2888.6 s; model load: 4.99 s.
- PyTorch peak allocated/reserved: 744.0 / 796.0 MiB (excludes driver and other processes).
- Per-file allocated latency: {"count": 5036, "p50": 0.5592986250012473, "p95": 0.9754497499980062, "max": 1.576327750000928}
- Per-sample sum allocated ASR: {"count": 1545, "p50": 1.4987319499996374, "p95": 5.360200140009691, "max": 8.49868324999261}
- Samples with allocated ASR >30 s: 0; incomplete sample timing: 0.
- Global duration-sorted batches cross samples. Allocations are not per-conversation wall time. ASR-only timings do not certify the competition end-to-end deadline or A30 performance.

## Private artifacts (not included in Git)

The following paths are relative to the private `LIFEMATE_SPEECH_WORK_DIR`:

- `train_full_gpu_2026-11-07/`: transcripts.jsonl, errors.jsonl, run_manifest.json, device_profile.json.
- `eval_train_full_gpu_2026-11-07/`: metrics.json, per_utterance.jsonl, worst_samples.jsonl (100), asr_report.md, errors.jsonl.
- The same evaluation directory contains entity_examples.md, entity_rules_and_examples.json, parser_uncertainties.md/.jsonl and full_validation.json.

This committed report contains aggregate statistics and hashes only. It excludes raw audio, reference/hypothesis text, per-sample records and model/cache files.

## Remaining work under team plan

- Human review/listen to worst samples and uncertain entity parses; do not correct hypotheses using gold.
- Align Speech handoff with frozen DATA IDs and the real OPS transcriber adapter.
- B1/S0_AUDIO and paired B0 comparison still require LLM + ToolEnv evaluation on the frozen validation set.
- Public transcription and B3-A/offline target-hardware rehearsal are separate, not run here.
- Detailed run artifacts contain private competition data; share through team-private storage, not Git. This aggregate report is the public-source summary.
