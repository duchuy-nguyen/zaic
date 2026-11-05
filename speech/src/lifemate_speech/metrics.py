"""Case/punctuation-sensitive WER/CER after the sole shared normalize().

WER tokens are whitespace-delimited Vietnamese syllable-like units, not linguistic
words. CER includes normalized spaces and punctuation. Empty reference: score 0
for empty hypothesis, otherwise edit count / max(1, reference length).
Tie-breaking: match/substitution, deletion, insertion. Missing/error hypotheses
are empty strings (deletions), never excluded from accuracy denominators.
"""
from .normalize import normalize


def edit_counts(reference, hypothesis):
    """Levenshtein optimal path counts S/D/I using O(len(hypothesis)) memory."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis)+1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            diagonal = previous[j-1]
            if ref == hyp:
                candidates = [diagonal]
            else:
                candidates = [(diagonal[0]+1, diagonal[1]+1, diagonal[2], diagonal[3])]
            deletion, insertion = previous[j], current[j-1]
            candidates += [(deletion[0]+1, deletion[1], deletion[2]+1, deletion[3]),
                           (insertion[0]+1, insertion[1], insertion[2], insertion[3]+1)]
            current.append(min(candidates, key=lambda item: item[0]))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return dict(S=substitutions, D=deletions, I=insertions, distance=distance)


def score_utterance(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    words, chars = edit_counts(ref.split(), hyp.split()), edit_counts(ref, hyp)
    return dict(reference_norm=ref, hypothesis_norm=hyp, wer=words['distance']/max(1,len(ref.split())),
                cer=chars['distance']/max(1,len(ref)), n_ref_words=len(ref.split()), n_ref_chars=len(ref),
                edit_ops={k: words[k] for k in ('S','D','I')}, char_edit_ops={k: chars[k] for k in ('S','D','I')})


def aggregate(rows):
    n = len(rows)
    word_edits = sum(sum(r['edit_ops'].values()) for r in rows)
    char_edits = sum(sum(r['char_edit_ops'].values()) for r in rows)
    word_count, char_count = sum(r['n_ref_words'] for r in rows), sum(r['n_ref_chars'] for r in rows)
    return dict(n_utterances=n, n_ref_words=word_count, n_ref_chars=char_count,
                word_edit_ops={k: sum(r['edit_ops'][k] for r in rows) for k in ('S','D','I')},
                char_edit_ops={k: sum(r['char_edit_ops'][k] for r in rows) for k in ('S','D','I')},
                wer_micro=word_edits/max(1,word_count) if n else None,
                cer_micro=char_edits/max(1,char_count) if n else None,
                wer_macro=sum(r['wer'] for r in rows)/n if n else None,
                cer_macro=sum(r['cer'] for r in rows)/n if n else None)


def percentiles(values):
    values = sorted(values)
    def at(q):
        pos = (len(values)-1)*q
        lo = int(pos)
        hi = min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(pos-lo) if values else None
    return dict(count=len(values), p50=at(.5) if values else None,
                p95=at(.95) if values else None, max=max(values) if values else None)
