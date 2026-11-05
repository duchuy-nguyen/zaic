"""Post-inference diagnostics only; never imported by the agent or ASR provider."""
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata

from .data import read_rows
from .references import PATH as REFERENCE_PATH
from .scoring import exact


def tokens(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).lower(), flags=re.UNICODE)


def edit_distance(reference, hypothesis):
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


@lru_cache(maxsize=50000)
def transcript_errors(reference, hypothesis):
    ref, hyp = tokens(reference), tokens(hypothesis)
    ref_chars, hyp_chars = ''.join(ref), ''.join(hyp)
    return {"word_edits": edit_distance(ref, hyp), "reference_words": len(ref),
            "character_edits": edit_distance(ref_chars, hyp_chars), "reference_characters": len(ref_chars)}


def diagnostics(directory: Path):
    rows = {r["id"]: r for r in read_rows()}
    references = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))["references"]
    traces = [json.loads(line) for line in (directory / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    totals = {"word_edits": 0, "reference_words": 0, "character_edits": 0, "reference_characters": 0}
    exact_sequences = nonempty_state_correct = nonempty_gold = 0
    missing_current = []
    nonempty_reasoning = set()
    for record in traces:
        sample_id = record["id"]
        ref = references[sample_id]
        exact_sequences += exact(record["tool_calls"], ref["tool_calls"])
        if ref["tool_calls"]:
            nonempty_gold += 1
            nonempty_state_correct += record["state_hash"] == ref["state_hash"]
        for step in record["trace"]:
            message = step["response"]["choices"][0]["message"]
            if message.get("reasoning_content") or "<think>" in (message.get("content") or ""):
                nonempty_reasoning.add(sample_id)
        current = rows[sample_id]["audio"]
        if not record.get("asr") or current not in record["asr"]["transcripts"]:
            missing_current.append(sample_id)
            continue
        errors = transcript_errors(rows[sample_id]["audio_text"], record["asr"]["transcripts"][current])
        for key in totals:
            totals[key] += errors[key]
    return {"samples": len(traces), "exact_tool_sequence_accuracy": exact_sequences/len(traces),
            "nonempty_gold_state_accuracy": nonempty_state_correct/nonempty_gold if nonempty_gold else None,
            "nonempty_gold_samples": nonempty_gold,
            "current_asr_wer": totals["word_edits"]/totals["reference_words"] if totals["reference_words"] else None,
            "current_asr_cer": totals["character_edits"]/totals["reference_characters"] if totals["reference_characters"] else None,
            "asr_error_counts": totals, "missing_current_transcripts": missing_current,
            "reasoning_samples": sorted(nonempty_reasoning),
            "asr_definition": "Current-turn corpus micro edit rates; NFC/lowercase Unicode word tokens, punctuation discarded, accents retained; CER removes whitespace. Vietnamese WER uses whitespace units, often syllables. Not BTC metrics. Missing transcripts reported explicitly, not silently scored as perfect."}
