from collections import Counter, defaultdict
from typing import List, Tuple, Union

import numpy as np

from fuzzy_search.tokenization.token import Doc, Token

_SMALL = 1e-20


class Ngram:

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.size = len(self.tokens)
        self.n = self.size
        self.string = ' '.join(tokens)
        self.head_string = ' '.join(tokens[:-1])
        self.tail_string = tokens[-1]


def doc_to_string_tokens(doc: Union[Doc, List[Token], List[str]]):
    if isinstance(doc, Doc):
        tokens = [token.n for token in doc.tokens]
    elif isinstance(doc, list) and isinstance(doc[0], Token):
        tokens = [token.n for token in doc]
    else:
        tokens = doc
    return tokens


def make_token_ngrams(doc: Union[Doc, List[Token], List[str]], ngram_size: int):
    tokens = doc_to_string_tokens(doc)
    # Add start and end symbols
    tokens = ['<s>'] * (ngram_size - 1) + tokens + ['</s>'] * (ngram_size - 1)
    num_tokens = len(tokens)
    for ti in range(len(tokens) - (ngram_size - 1)):
        # if ti + ngram_size <= num_tokens:
        ngram = tokens[ti:ti + ngram_size]
        yield ngram
    return None


def check_lambdas(lambdas: List[float], ngram_size: int):
    if lambdas is None or len(lambdas) != ngram_size:
        raise ValueError(f"smoothing a {ngram_size}-gram with "
                         f"interpolation requires {ngram_size} lambda values")
    elif any(isinstance(l, float) is False for l in lambdas):
        raise TypeError(f"lambdas must be floats, not {[type(l) for l in lambdas]}")
    elif sum(lambdas) != 1.0:
        raise ValueError(f"lambda values must sum to 1.0, current values sum to {sum(lambdas)}")


class NgramFreq:

    def __init__(self, max_ngram_size: int = 3):
        self.ngram_freq = defaultdict(Counter)
        self.max_ngram_size = max_ngram_size
        self.total_ngram_tokens = Counter()
        self.total_ngram_types = Counter()
        self.num_docs = 0
        self.start_tokens = {' '.join(['<s>'] * i) for i in range(1, max_ngram_size+1)}
        self.end_tokens = {' '.join(['</s>'] * i) for i in range(1, max_ngram_size+1)}

    def __getitem__(self, item):
        if item in self.start_tokens or item in self.end_tokens:
            return self.num_docs
        ngram_size = item.count(' ') + 1
        if ngram_size > self.max_ngram_size:
            return 0
        return self.ngram_freq[ngram_size][item] if item in self.ngram_freq[ngram_size] else 0

    def count_ngrams(self, docs: List[Doc]):
        self.num_docs += len(docs)
        for di, doc in enumerate(docs):
            for ngram_size in range(1, self.max_ngram_size + 1):
                self.ngram_freq[ngram_size].update([' '.join(ngram) for ngram in make_token_ngrams(doc, ngram_size)])
        for ngram_size in range(1, self.max_ngram_size + 1):
            self.total_ngram_tokens[ngram_size] = sum(self.ngram_freq[ngram_size].values())
            self.total_ngram_types[ngram_size] = len(self.ngram_freq[ngram_size])

    @property
    def vocab_size(self):
        return len(self.ngram_freq[1])

    def has_freq(self, term: str):
        ngram_size = term.count(' ') + 1
        return self.ngram_freq[ngram_size][term] if term in self.ngram_freq[ngram_size] else 0

    def has_prob(self, term: str, smoothing: str = None):
        ngram_freq = self.has_freq(term)
        ngram_size = term.count(' ') + 1
        if smoothing == 'laplace':
            return (ngram_freq + 1) / (self.total_tokens(ngram_size) + self.vocab_size)
        else:
            return ngram_freq / self.total_tokens(ngram_size)

    def has_conditional_prob(self, ngram_string: str, smoothing: str = None,
                             lambdas: List[float] = None, k: float = 1.0):
        tokens = ngram_string.split(' ')
        ngram_size = len(tokens)
        if smoothing is None:
            return self._has_unsmoothed_conditional_prob(ngram_string, tokens, ngram_size)
        if smoothing == 'laplace':
            return self._has_laplace_smoothed_conditional_prob(ngram_string, tokens, ngram_size, k=k)
        elif smoothing == 'interpolation':
            return self._has_interpolated_conditional_prob(ngram_string, tokens, ngram_size, lambdas)
        else:
            raise ValueError(f"invalid smoothing value, must be one of None, 'laplace' or 'interpolation'.")

    def _get_head_count(self, tokens: List[str], ngram_size: int):
        if ngram_size == 1:
            return self.total_ngram_tokens[1]
        else:
            head = ' '.join(tokens[:-1])
            return self.__getitem__(head)

    def _has_unsmoothed_conditional_prob(self, ngram_string: str, tokens: List[str], ngram_size: int):
        ngram_count = self.__getitem__(ngram_string)
        head_count = self._get_head_count(tokens, ngram_size)
        if head_count == 0:
            return 0.0
            # raise ValueError(f"ngram '{ngram_string}' has head count of 0")
        return ngram_count / head_count

    def _has_laplace_smoothed_conditional_prob(self, ngram_string: str, tokens: List[str],
                                               ngram_size: int, k: float = 1.0):
        ngram_count = self.__getitem__(ngram_string)
        head_count = self._get_head_count(tokens, ngram_size)
        if head_count == 0:
            return 0.0
            # raise ValueError(f"ngram '{ngram_string}' has head count of 0")
        return (ngram_count + k) / (head_count + k * self.vocab_size)

    def _has_interpolated_conditional_prob(self, ngram_string: str, tokens: List[str],
                                           ngram_size: int, lambdas: List[float]):
        check_lambdas(lambdas, ngram_size)
        cond_prob = 0.0
        try:
            for n in range(ngram_size):
                tail_ngram = ' '.join(tokens[-n:])
                lambda_n = lambdas[n]
                tail_prob = self._has_unsmoothed_conditional_prob(tail_ngram, tokens[-n:], ngram_size - n)
                cond_prob += lambda_n * tail_prob
        except IndexError:
            print(f"ngram_size: {ngram_size}")
            raise
        return cond_prob

    def total_tokens(self, ngram_size: int = 1):
        return self.total_ngram_tokens[ngram_size]

    def total_types(self, ngram_size: int = 1):
        return self.total_ngram_types[ngram_size]


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
