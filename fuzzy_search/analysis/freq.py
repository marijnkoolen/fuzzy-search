"""Token ngram frequency counting and probability estimation, plus log-likelihood-ratio
keyword/word comparison statistics.

This module provides ``NgramFreq`` for counting word ngrams across a collection of
documents and computing (optionally smoothed) ngram probabilities and conditional
probabilities, along with functions for computing the log-likelihood ratio (LLR) and
percentage-difference statistics that compare a word's frequency between a target and
reference corpus.
"""

from collections import Counter, defaultdict
from typing import List, Tuple, Union

import numpy as np

from fuzzy_search.tokenization.token import Doc, Token

_SMALL = 1e-20


class Ngram:
    """Represents a word ngram, exposing both the full ngram and its head/tail parts.

    Attributes:
        tokens (List[str]): The ngram's token strings.
        size (int): The number of tokens in the ngram.
        n (int): Alias of ``size``.
        string (str): The ngram tokens joined with spaces.
        head_string (str): All tokens except the last, joined with spaces (the
            "history" used for conditional probability estimation).
        tail_string (str): The last token in the ngram.
    """

    def __init__(self, tokens: List[str]):
        """Initializes an Ngram from a list of token strings.

        Args:
            tokens (List[str]): The tokens making up the ngram.
        """
        self.tokens = tokens
        self.size = len(self.tokens)
        self.n = self.size
        self.string = ' '.join(tokens)
        self.head_string = ' '.join(tokens[:-1])
        self.tail_string = tokens[-1]


def doc_to_string_tokens(doc: Union[Doc, List[Token], List[str]]):
    """Normalizes a document representation into a plain list of normalised token strings.

    Args:
        doc (Union[Doc, List[Token], List[str]]): A Doc, a list of Token objects, or
            a list of strings.

    Returns:
        List[str]: The normalised token strings of the document.
    """
    if isinstance(doc, Doc):
        tokens = [token.n for token in doc.tokens]
    elif isinstance(doc, list) and isinstance(doc[0], Token):
        tokens = [token.n for token in doc]
    else:
        tokens = doc
    return tokens


def make_token_ngrams(doc: Union[Doc, List[Token], List[str]], ngram_size: int):
    """Generates word ngrams of a given size from a document, padded with start/end symbols.

    The document's tokens are padded with ``ngram_size - 1`` ``<s>`` start symbols and
    ``</s>`` end symbols so that ngrams overlapping the document boundaries are also
    produced (e.g. for a trigram model, this yields ngrams like ``['<s>', '<s>', tok1]``).

    Args:
        doc (Union[Doc, List[Token], List[str]]): The document to extract ngrams from.
        ngram_size (int): The number of tokens per ngram.

    Yields:
        List[str]: Each successive ngram (as a list of token strings) in the document.
    """
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
    """Validates interpolation lambda weights for ngram smoothing.

    Args:
        lambdas (List[float]): The interpolation weights, one per ngram order from 1 up
            to ``ngram_size``.
        ngram_size (int): The ngram order being smoothed.

    Raises:
        ValueError: If ``lambdas`` is missing, has the wrong length, or its values don't
            sum to 1.0.
        TypeError: If any value in ``lambdas`` is not a float.
    """
    if lambdas is None or len(lambdas) != ngram_size:
        raise ValueError(f"smoothing a {ngram_size}-gram with "
                         f"interpolation requires {ngram_size} lambda values")
    elif any(isinstance(l, float) is False for l in lambdas):
        raise TypeError(f"lambdas must be floats, not {[type(l) for l in lambdas]}")
    elif sum(lambdas) != 1.0:
        raise ValueError(f"lambda values must sum to 1.0, current values sum to {sum(lambdas)}")


class NgramFreq:
    """Counts word ngram frequencies (up to a maximum order) across a collection of documents
    and provides (smoothed) probability and conditional probability estimation.

    Attributes:
        ngram_freq (Dict[int, Counter]): Maps ngram order -> Counter of ngram string -> frequency.
        max_ngram_size (int): The highest ngram order that is counted.
        total_ngram_tokens (Counter): Maps ngram order -> total ngram token count.
        total_ngram_types (Counter): Maps ngram order -> number of distinct ngram types.
        num_docs (int): The number of documents counted so far.
        start_tokens (Set[str]): The padded ``<s>`` boundary ngram strings, for each order.
        end_tokens (Set[str]): The padded ``</s>`` boundary ngram strings, for each order.
    """

    def __init__(self, max_ngram_size: int = 3):
        """Initializes the NgramFreq counter.

        Args:
            max_ngram_size (int, optional): The highest ngram order to count, e.g. 3 for
                unigrams, bigrams and trigrams. Defaults to 3.
        """
        self.ngram_freq = defaultdict(Counter)
        self.max_ngram_size = max_ngram_size
        self.total_ngram_tokens = Counter()
        self.total_ngram_types = Counter()
        self.num_docs = 0
        self.start_tokens = {' '.join(['<s>'] * i) for i in range(1, max_ngram_size+1)}
        self.end_tokens = {' '.join(['</s>'] * i) for i in range(1, max_ngram_size+1)}

    def __getitem__(self, item):
        """Returns the frequency of an ngram string (boundary ngrams return ``num_docs``)."""
        if item in self.start_tokens or item in self.end_tokens:
            return self.num_docs
        ngram_size = item.count(' ') + 1
        if ngram_size > self.max_ngram_size:
            return 0
        return self.ngram_freq[ngram_size][item] if item in self.ngram_freq[ngram_size] else 0

    def count_ngrams(self, docs: List[Doc]):
        """Counts ngrams (of all orders up to ``max_ngram_size``) across a list of documents,
        updating the running totals.

        Args:
            docs (List[Doc]): The documents to count ngrams in.
        """
        self.num_docs += len(docs)
        for di, doc in enumerate(docs):
            for ngram_size in range(1, self.max_ngram_size + 1):
                self.ngram_freq[ngram_size].update([' '.join(ngram) for ngram in make_token_ngrams(doc, ngram_size)])
        for ngram_size in range(1, self.max_ngram_size + 1):
            self.total_ngram_tokens[ngram_size] = sum(self.ngram_freq[ngram_size].values())
            self.total_ngram_types[ngram_size] = len(self.ngram_freq[ngram_size])

    @property
    def vocab_size(self):
        """int: The number of distinct unigram types (the vocabulary size)."""
        return len(self.ngram_freq[1])

    def has_freq(self, term: str):
        """Returns the frequency of an ngram string (without special-casing boundary tokens).

        Args:
            term (str): The (space-joined) ngram string to look up.

        Returns:
            int: The ngram's frequency, or 0 if unseen.
        """
        ngram_size = term.count(' ') + 1
        return self.ngram_freq[ngram_size][term] if term in self.ngram_freq[ngram_size] else 0

    def has_prob(self, term: str, smoothing: str = None):
        """Computes the (optionally Laplace-smoothed) probability of an ngram string.

        Args:
            term (str): The (space-joined) ngram string.
            smoothing (str, optional): If ``'laplace'``, apply add-one smoothing over the
                vocabulary; otherwise compute the unsmoothed maximum-likelihood probability.

        Returns:
            float: The estimated probability of the ngram.
        """
        ngram_freq = self.has_freq(term)
        ngram_size = term.count(' ') + 1
        if smoothing == 'laplace':
            return (ngram_freq + 1) / (self.total_tokens(ngram_size) + self.vocab_size)
        else:
            return ngram_freq / self.total_tokens(ngram_size)

    def has_conditional_prob(self, ngram_string: str, smoothing: str = None,
                             lambdas: List[float] = None, k: float = 1.0):
        """Computes the conditional probability of an ngram's last token given its preceding tokens.

        Args:
            ngram_string (str): The (space-joined) ngram string.
            smoothing (str, optional): One of None (unsmoothed), ``'laplace'`` (add-k
                smoothing), or ``'interpolation'`` (linear interpolation across orders).
            lambdas (List[float], optional): Interpolation weights, required when
                ``smoothing == 'interpolation'``.
            k (float, optional): The add-k smoothing constant, used when
                ``smoothing == 'laplace'``. Defaults to 1.0.

        Returns:
            float: The estimated conditional probability.

        Raises:
            ValueError: If ``smoothing`` is not one of the supported values.
        """
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
        """Returns the frequency of the ngram's head (all but the last token), or the
        total unigram count when ``ngram_size`` is 1."""
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
                print(f"\tn: {n}\tlambda: {lambda_n}\ttail_ngram: {tail_ngram}\ttail_prob: {tail_prob}")
                cond_prob += lambda_n * tail_prob
        except IndexError:
            print(f"ngram_size: {ngram_size}")
            raise
        if cond_prob == 0.0:
            cond_prob = 1 / self.total_tokens(1)
            print(f"cond_prob == 0.0\tself.total_tokens(1): {self.total_tokens(1)}\tcond_prob: {cond_prob}")
        return cond_prob

    def total_tokens(self, ngram_size: int = 1):
        """Returns the total count of ngram tokens of a given order seen so far."""
        return self.total_ngram_tokens[ngram_size]

    def total_types(self, ngram_size: int = 1):
        """Returns the number of distinct ngram types of a given order seen so far."""
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
    """Computes the log-likelihood ratio (G2 statistic) from an observed 2x2 contingency table.

    Args:
        observed (np.array): A 2x2 contingency table, as produced by
            :func:`get_observed_from_counter`.
        include_direction (bool, optional): If True, also return whether the target
            count is higher ('more') or lower ('less') than expected under independence.

    Returns:
        Union[float, Tuple[float, str]]: The log-likelihood ratio, optionally paired
        with the direction string.
    """
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
    """Computes the relative percentage difference in a token's frequency fraction between
    a target and a reference corpus.

    Args:
        token: The token to compare.
        target_counter (Counter): Token frequencies in the target corpus.
        target_total (int): Total token count in the target corpus.
        reference_counter (Counter): Token frequencies in the reference corpus.
        reference_total (int): Total token count in the reference corpus.

    Returns:
        float: The percentage difference of the target fraction relative to the reference
        fraction, or ``float('inf')`` if the token does not occur in the reference corpus.
    """
    target_freq = target_counter[token] if token in target_counter else 0
    ref_freq = reference_counter[token] if token in reference_counter else 0
    if ref_freq == 0:
        return float('inf')
    target_frac = target_freq / target_total
    ref_frac = ref_freq / reference_total
    return (target_frac - ref_frac) / ref_frac
