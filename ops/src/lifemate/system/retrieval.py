"""Deterministic CPU lexical retrieval; disabled by default, no external models."""
from collections import Counter
from copy import deepcopy
import math
import re
import unicodedata


def features(text):
    words = re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())
    # Character features retain some robustness to spelling/ASR variation.
    return Counter(words + ['#' + w[i:i+3] for w in words for i in range(len(w)-2)])


def similarity(query, document):
    q, d = features(query), features(document)
    if not q or not d:
        return 0.0
    return sum(v*d[k] for k, v in q.items()) / math.sqrt(
        sum(v*v for v in q.values()) * sum(v*v for v in d.values()))


class ToolSelector:
    """Select from already permitted native schemas; explicit dependency closure.

    top_k=0 preserves the full catalog. A repair or zero lexical evidence opens
    the full permitted set. Similarity is not an OOS classifier.
    """
    def __init__(self, top_k=0, dependencies=None):
        if type(top_k) is not int or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer')
        self.top_k = top_k
        self.dependencies = deepcopy(dependencies or {})
        if any(not isinstance(k, str) or not isinstance(v, (tuple, list))
               or any(not isinstance(n, str) for n in v)
               for k, v in self.dependencies.items()):
            raise ValueError('Dependencies must map names to tool name lists')

    def select(self, query, tools, *, repair=False):
        if not self.top_k or repair:
            return deepcopy(tools)
        by_name = {t['function']['name']: t for t in tools}
        scored = []
        for name, tool in by_name.items():
            f = tool['function']
            document = name + ' ' + f.get('description', '') + ' ' + ' '.join(
                f['parameters'].get('properties', {}))
            scored.append((similarity(query, document), name))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if not scored or scored[0][0] <= 0:
            return deepcopy(tools)
        selected = {n for _, n in scored[:self.top_k]} | ({'sys.ask_missing_info'} & by_name.keys())
        pending = list(selected)
        while pending:
            for dependency in self.dependencies.get(pending.pop(), ()):
                if dependency in by_name and dependency not in selected:
                    selected.add(dependency)
                    pending.append(dependency)
        return [deepcopy(by_name[n]) for n in sorted(selected)]


class ExampleBank:
    """Train-fit procedural demonstrations, not executable calls/entity IDs.

    The caller owns delexicalization and an honest frozen split manifest. No
    validation/public/private corpus loader or automatic labeling is provided.
    """
    def __init__(self, examples, *, fit_ids, split_hash, top_k=0):
        if not isinstance(split_hash, str) or not split_hash.strip():
            raise ValueError('Frozen split hash required')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('Invalid example count')
        self.split_hash, self.top_k = split_hash, top_k
        self.examples = deepcopy(list(examples))
        seen = set()
        for e in self.examples:
            if (set(e) != {'id', 'utterance', 'guidance', 'tools'}
                    or e['id'] not in fit_ids or e['id'] in seen
                    or not all(isinstance(e[k], str) and e[k] for k in ('id', 'utterance', 'guidance'))
                    or type(e['tools']) is not list
                    or any(not isinstance(t, str) for t in e['tools'])):
                raise ValueError('Example must be unique, train-fit and use the procedural schema')
            seen.add(e['id'])

    def retrieve(self, query, permitted):
        scored = [(similarity(query, e['utterance']), e) for e in self.examples
                  if set(e['tools']) <= permitted]
        scored.sort(key=lambda x: (-x[0], x[1]['id']))
        return [deepcopy(e) for score, e in scored[:self.top_k] if score > 0]
