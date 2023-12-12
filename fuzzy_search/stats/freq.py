from collections import Counter, defaultdict
from typing import List, Tuple, Union

import numpy as np

from fuzzy_search.tokenization.token import Doc

_SMALL = 1e-20


class NgramFreq:

    def __init__(self, max_ngram_size: int = 3):
        self.ngram_freq = defaultdict(Counter)
        self.max_ngram_size = max_ngram_size

    def __getitem__(self, item):
        ngram_size = item.count(' ') + 1
        if ngram_size > self.max_ngram_size:
            return 0
        return self.ngram_freq[ngram_size][item] if item in self.ngram_freq[ngram_size] else 0

    def count_ngrams(self, docs: List[Doc]):
        for doc in docs:
            num_tokens = len(doc)
            for ti in range(len(doc)):
                for ngram_size in range(1, self.max_ngram_size + 1):
                    if ti + ngram_size <= num_tokens:
                        tokens = doc.tokens[ti:ti+ngram_size]
                        ngram = ' '.join([token.n for token in tokens])
                        self.ngram_freq[ngram_size].update([ngram])

    def has_freq(self, term: str):
        ngram_size = term.count(' ') + 1
        return self.ngram_freq[ngram_size][term] if term in self.ngram_freq[ngram_size] else 0

    def total_tokens(self, ngram_size: int = 1):
        return sum(self.ngram_freq[ngram_size].values())

    def total_types(self, ngram_size: int = 1):
        return len(self.ngram_freq[ngram_size])


def get_observed_from_counter(token: str, target_counter: Counter, target_total: int,
                              reference_counter: Counter, reference_total: int):
    """Computes the contingency table of the observed values given a target token, and
    target and reference analysers and counters."""
    # a: word in target corpus
    t_target = target_counter[token] if token in target_counter else 0
    # b: word in ref corpus
    t_ref = reference_counter[token] if token in reference_counter else 0
    # c: other words in target corpus
    nt_target = target_total - t_target
    # d: other words in ref corpus
    nt_ref = reference_total - t_ref

    observed = np.array([
        [t_target, t_ref],
        [nt_target, nt_ref]
    ])
    return observed


def compute_expected(observed: np.array) -> np.array:
    """Computes the contingency table of the expected values given a contingency table
    of the observed values."""
    expected = np.array([
        [
            observed[0, :].sum() * observed[:, 0].sum() / observed.sum(),
            observed[0, :].sum() * observed[:, 1].sum() / observed.sum()
        ],
        [
            observed[1, :].sum() * observed[:, 0].sum() / observed.sum(),
            observed[1, :].sum() * observed[:, 1].sum() / observed.sum()
        ]
    ])
    return expected


def compute_llr_from_observed(observed: np.array,
                              include_direction: bool = False) -> Union[float, Tuple[float, str]]:
    sum_likelihood = 0
    expected = compute_expected(observed)
    for i in [0, 1]:
        for j in [0, 1]:
            sum_likelihood += observed[i, j] * np.log((observed[i, j] + _SMALL) / (expected[i, j] + _SMALL))
    if include_direction is True:
        return 2 * sum_likelihood, 'more' if observed[0, 0] > expected[0, 0] else 'less'
    else:
        return 2 * sum_likelihood


def compute_llr(token: str, target_counter: Counter, target_total: int,
                reference_counter: Counter, reference_total: int,
                include_direction: bool = False) -> Tuple[float, str]:
    """Computes the log likelihood ratio for given a target token, and target and
    reference analysers and counters."""
    observed = get_observed_from_counter(token, target_counter, target_total, reference_counter,
                                         reference_total)
    return compute_llr_from_observed(observed, include_direction=include_direction)


def compute_percentage_diff(token, target_counter, target_total, reference_counter, reference_total):
    target_freq = target_counter[token] if token in target_counter else 0
    ref_freq = reference_counter[token] if token in reference_counter else 0
    if ref_freq == 0:
        return float('inf')
    target_frac = target_freq / target_total
    ref_frac = ref_freq / reference_total
    return (target_frac - ref_frac) / ref_frac
