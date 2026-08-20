"""Term similarity utilities: Levenshtein-based begin/end similarity, term skip-cooccurrence
counting, and an efficient skipgram-based cosine-similarity index for finding terms similar
to a query term.

The centerpiece is ``SkipgramSimilarity``, which indexes a vocabulary of terms by their
character skipgram frequency vectors, bucketed by term length, and uses sparse
matrix-vector products to retrieve the most similar indexed terms to a query term.
"""

import math
from collections import defaultdict
from collections import Counter
from typing import Dict, Generator, Iterable, List, Tuple, Union

import numpy as np
import scipy.sparse as sp

from fuzzy_search.tokenization.string import score_levenshtein_distance
from fuzzy_search.tokenization.string import text2skipgrams
from fuzzy_search.tokenization.vocabulary import Vocabulary


def vector_length(skipgram_freq):
    """Computes the Euclidean length of a skipgram frequency vector.

    Args:
        skipgram_freq: A mapping (e.g. dict or Counter) of skipgram -> frequency.

    Returns:
        float: The square root of the sum of squared frequencies.
    """
    return math.sqrt(sum([skipgram_freq[skip] ** 2 for skip in skipgram_freq]))


def get_skip_coocs(seq_ids: List[str], skip_size: int = 0) -> Generator[Tuple[int, int], None, None]:
    """Generates all (current, following) id pairs within a skip window of a sequence.

    Args:
        seq_ids (List[str]): A sequence of (term) identifiers.
        skip_size (int, optional): The number of extra positions beyond the immediate
            neighbour to pair with each element. Defaults to 0 (only adjacent pairs).

    Yields:
        Tuple[int, int]: Each (current_id, following_id) co-occurrence pair.
    """
    for ci, curr_id in enumerate(seq_ids):
        for next_id in seq_ids[ci + 1: ci + 2 + skip_size]:
            yield curr_id, next_id


def get_min_length(phrase1: str, phrase2: str, begin_length: int) -> int:
    """Returns the smallest of the two phrase lengths and a requested length."""
    return min([len(phrase1), len(phrase2), begin_length])


def get_begin_sim(phrase1: str, phrase2: str, begin_length: int) -> float:
    """Computes the Levenshtein-based similarity of the first ``begin_length`` characters
    of two phrases.

    Args:
        phrase1 (str): The first phrase.
        phrase2 (str): The second phrase.
        begin_length (int): The number of leading characters to compare (clamped to the
            shorter phrase's length).

    Returns:
        float: 1 minus the normalised Levenshtein distance between the two prefixes.
    """
    if min(len(phrase1), len(phrase2)) < begin_length:
        begin_length = min(len(phrase1), len(phrase2))
    begin1 = phrase1[:begin_length]
    begin2 = phrase2[:begin_length]
    return 1 - (score_levenshtein_distance(begin1, begin2) / begin_length)


def get_end_sim(phrase1: str, phrase2: str, end_length: int) -> float:
    """Computes the Levenshtein-based similarity of the last ``end_length`` characters
    of two phrases.

    Args:
        phrase1 (str): The first phrase.
        phrase2 (str): The second phrase.
        end_length (int): The number of trailing characters to compare (clamped to the
            shorter phrase's length).

    Returns:
        float: 1 minus the normalised Levenshtein distance between the two suffixes.
    """
    if min(len(phrase1), len(phrase2)) < end_length:
        end_length = min(len(phrase1), len(phrase2))
    end1 = phrase1[-end_length:]
    end2 = phrase2[-end_length:]
    return 1 - (score_levenshtein_distance(end1, end2) / end_length)


class SkipCooccurrence:
    """Counts the co-occurrence frequency of word skipgrams within sentences.

    Attributes:
        cooc_freq (Dict[Tuple[int, int], int]): Maps (term_id, term_id) co-occurrence
            pairs to their observed frequency.
        vocabulary (Vocabulary): Maps terms to identifiers used in ``cooc_freq``.
        skip_size (int): The default maximum number of skips allowed between
            co-occurring terms.
    """

    def __init__(self, vocabulary: Vocabulary, skip_size: int = 1, sentences: Iterable[List[str]] = None):
        """Initializes the SkipCooccurrence counter.

        Args:
            vocabulary (Vocabulary): The vocabulary used to map terms to identifiers.
            skip_size (int, optional): The maximum number of skips allowed between
                co-occurring terms. Defaults to 1.
            sentences (Iterable[List[str]], optional): If given, immediately count
                co-occurrences for these sentences.
        """
        self.cooc_freq = defaultdict(int)
        self.vocabulary = vocabulary
        self.skip_size: int = skip_size
        if sentences is not None:
            self.calculate_skip_cooccurrences(sentences)

    def calculate_skip_cooccurrences(self, sentences: Iterable[List[str]], skip_size: int = 0):
        """Count the frequency of term (skip) co-occurrences for a given list of sentences.

        :param sentences: a list of sentences, where each sentence is itself a list of term tokens
        :type sentences: Iterable[List[str]
        :param skip_size: the maximum number of skips to allow between co-occurring terms
        :type skip_size: int
        """
        for sent in sentences:
            seq_ids = [self.vocabulary.term2id(t) for t in sent]
            self.cooc_freq.update(get_skip_coocs(seq_ids, skip_size=skip_size))

    def _cooc_ids2terms(self, cooc_ids: Tuple[int, int]) -> Tuple[str, str]:
        """Converts a pair of term identifiers back into their term strings."""
        id1, id2 = cooc_ids
        return self.vocabulary.id2term(id1), self.vocabulary.id2term(id2)

    def get_term_coocs(self, term: str) -> Union[None, Generator[Tuple[str, str], None, None]]:
        """Yields all co-occurring term pairs and their frequency for a given term.

        Args:
            term (str): The term to look up co-occurrences for.

        Yields:
            Tuple[Tuple[str, str], int]: Each (term pair, frequency) where ``term``
            is one of the two terms in the pair.

        Returns:
            None: If ``term`` is not in the vocabulary.
        """
        term_id = self.vocabulary.term2id(term)
        if term_id is None:
            return None
        for cooc_ids in self.cooc_freq:
            if term_id in cooc_ids:
                yield self._cooc_ids2terms(cooc_ids), self.cooc_freq[cooc_ids]


def is_close_distance_keyword_pair(keyword1: str, keyword2: str, max_distance_ratio: float,
                                   max_length_difference: int, max_distance: int) -> bool:
    """Checks whether two keywords are close enough to be considered near-duplicates.

    Two keywords are considered close if their length difference is within
    ``max_length_difference`` and their Levenshtein distance is both below
    ``max_distance`` and below ``max_distance_ratio`` relative to either keyword's length.

    Args:
        keyword1 (str): The first keyword.
        keyword2 (str): The second keyword.
        max_distance_ratio (float): The maximum allowed distance-to-length ratio.
        max_length_difference (int): The maximum allowed difference in keyword lengths.
        max_distance (int): The maximum allowed absolute Levenshtein distance.

    Returns:
        bool: True if the keywords are considered a close-distance pair.
    """
    if abs(len(keyword1) - len(keyword2)) > max_length_difference:
        return False
    distance = score_levenshtein_distance(keyword1, keyword2)
    if distance < max_distance and (
            distance / len(keyword1) < max_distance_ratio or distance / len(keyword2) < max_distance_ratio):
        return True
    return False


class KeywordList:
    """Indexes a list of keywords by length, to efficiently enumerate candidate pairs
    of keywords whose lengths are close enough to plausibly be near-duplicates.

    Attributes:
        len_keys (Dict[int, List[str]]): Maps keyword length to the keywords of that length.
        max_length_diff (int): The default maximum length difference used when generating
            candidate pairs.
        len_order (List[int]): The distinct keyword lengths, sorted ascending.
    """

    def __init__(self, keywords: List[str], max_length_diff: int):
        """Initializes the KeywordList by indexing keywords by their length.

        Args:
            keywords (List[str]): The keywords to index.
            max_length_diff (int): The default maximum length difference allowed when
                comparing keyword pairs.

        Raises:
            ValueError: If any keyword is not a string.
        """
        self.len_keys = defaultdict(list)
        self.max_length_diff = max_length_diff
        for ki, keyword in enumerate(keywords):
            if isinstance(keyword, str) is False:
                raise ValueError(f"keyword '{keyword}' at index {ki} is not of "
                                 f"type str but type {type(keyword)}")
            len_key = len(keyword)
            self.len_keys[len_key].append(keyword)
        self.len_order = sorted(self.len_keys.keys())

    def iterate_candidate_pairs(self):
        """Yields all candidate keyword pairs whose lengths differ by at most
        ``max_length_diff``, without yielding the same unordered pair twice.

        Yields:
            Tuple[str, str]: Each candidate (kw1, kw2) pair.
        """
        for len_key1 in self.len_order:
            for ki, kw1 in enumerate(self.len_keys[len_key1]):
                for len_key2 in range(len_key1, len_key1 + self.max_length_diff + 1):
                    start = ki + 1 if len_key2 == len_key1 else 0
                    for kw2 in self.len_keys[len_key2][start:]:
                        yield kw1, kw2

    def find_close_distance_keywords(self, max_distance_ratio: float = 0.3,
                                     max_length_diff: int = 3, max_distance: int = 10,
                                     ignorecase: bool = False) -> Dict[str, List[str]]:
        """TODO: should we make the arguments into a config?"""
        if max_length_diff is None:
            max_length_diff = self.max_length_diff
        close_distance_keywords = defaultdict(list)
        for keyword1, keyword2 in self.iterate_candidate_pairs():
            string1 = keyword1.lower() if ignorecase else keyword1
            string2 = keyword2.lower() if ignorecase else keyword2
            if is_close_distance_keyword_pair(string1, string2, max_distance_ratio=max_distance_ratio,
                                              max_length_difference=max_length_diff,
                                              max_distance=max_distance):
                close_distance_keywords[keyword1].append(keyword2)
                close_distance_keywords[keyword2].append(keyword1)
        return close_distance_keywords

    def find_closer_terms(self, candidate: str, keyword: str, close_terms: List[str]):
        """Finds which of a keyword's known close terms are even closer to a candidate
        string than the keyword itself.

        Args:
            candidate (str): The string to compare distances against.
            keyword (str): The reference keyword.
            close_terms (List[str]): Terms previously identified as close to ``keyword``.

        Returns:
            Dict[str, int]: Maps each close term that is nearer to ``candidate`` than
            ``keyword`` is, to its Levenshtein distance from ``candidate``.
        """
        closer_terms = {}
        keyword_distance = score_levenshtein_distance(keyword, candidate)
        # print("candidate:", candidate, "\tkeyword:", keyword)
        # print("keyword_distance", keyword_distance)
        for close_term in close_terms:
            close_term_distance = score_levenshtein_distance(close_term, candidate)
            # print("close_term:", close_term, "\tdistance:", close_term_distance)
            if close_term_distance < keyword_distance:
                closer_terms[close_term] = close_term_distance


class SkipgramSimilarity:
    """Indexes terms by their character skipgram frequency vectors and finds similar
    indexed terms for a query term via cosine similarity.

    Terms are bucketed by length (see ``max_length_diff``) and, within each length
    bucket, represented as columns of a sparse skipgram-by-term matrix. This lets a
    query be compared against only the terms within the relevant length range using a
    handful of sparse matrix-vector products, rather than scanning the whole vocabulary.

    Attributes:
        ngram_length (int): The number of characters per skipgram.
        skip_length (int): The maximum number of skipped characters per skipgram.
        vocabulary (Vocabulary): Maps indexed terms to term ids.
        skipgram_vocabulary (Vocabulary): Maps observed skipgram strings to skipgram ids.
        max_length_diff (int): The maximum term-length difference considered when
            searching for similar terms.
    """

    def __init__(self, ngram_length: int = 3, skip_length: int = 0, terms: List[str] = None,
                 max_length_diff: int = 2):
        """A class to index terms by their character skipgrams and to find similar terms for a given
        input term based on the cosine similarity of their skipgram overlap.

        :param ngram_length: the number of characters per ngram
        :type ngram_length: int
        :param skip_length: the maximum number of characters to skip
        :type skip_length: int
        :param terms: a list of terms
        :type terms: List[str]
        :param max_length_diff: the maximum difference in length between a search term and a term in the
            index to be considered a match. This is an efficiency parameter to reduce the number of candidate
            similar terms to ones that are roughly similar in length to the search term.
        :type max_length_diff: int
        """
        self.ngram_length = ngram_length
        self.skip_length = skip_length
        self.vocabulary = Vocabulary()
        self.skipgram_vocabulary = Vocabulary()
        self.max_length_diff = max_length_diff
        # term_id -> skipgram_id -> freq, updated incrementally as each term is indexed.
        self._term_skipgram_freq: Dict[int, Counter] = {}
        # term_length -> list of term_ids of that length, updated incrementally as
        # each term is indexed (mirrors the old skipgram_index[skipgram][term_length] grouping).
        self._term_ids_by_length: Dict[int, set] = defaultdict(set)
        # term_length -> sparse (num_skipgrams x num_terms_of_that_length) matrix.
        # Built lazily per length and only rebuilt when new terms of that length
        # have been indexed since it was last built (see self._dirty_lengths),
        # so indexing new terms never forces a rebuild of unrelated lengths.
        self._length_buckets: Dict[int, sp.csr_matrix] = {}
        # term_length -> np.ndarray mapping each bucket matrix column back to its term_id
        self._length_bucket_term_ids: Dict[int, np.ndarray] = {}
        # term_id -> vector length (sqrt of summed squared skipgram freqs), filled in
        # as each length bucket is (re)built.
        self._vector_length: Dict[int, float] = {}
        # lengths whose term_ids have changed since their bucket was last built
        self._dirty_lengths: set = set()
        if terms is not None:
            self.index_terms(terms)

    def _reset_index(self):
        """Clears the term and skipgram vocabularies and all internal index structures."""
        self.vocabulary.reset_index()
        self.skipgram_vocabulary.reset_index()
        self._term_skipgram_freq = {}
        self._term_ids_by_length = defaultdict(set)
        self._length_buckets = {}
        self._length_bucket_term_ids = {}
        self._vector_length = {}
        self._dirty_lengths = set()

    def index_terms(self, terms: List[str], reset_index: bool = True):
        """Make a frequency index of the skip grams for a given list of terms.
        By default, indexing is cumulative, that is, everytime you call index_terms
        with a list of terms, they are added to the index. Use 'reset_index=True' to
        reset the index before indexing the given terms.

        :param terms: a list of term to index
        :type terms: List[str]
        :param reset_index: whether to reset the index before indexing or to keep the existing index
        :type reset_index: bool
        """
        if reset_index is True:
            self._reset_index()
        self.vocabulary.add_terms(terms)
        for term in terms:
            self._index_term_skips(term)

    def _term_to_skip(self, term):
        """Returns a Counter of skipgram string frequencies for a term."""
        skip_gen = text2skipgrams(term, ngram_size=self.ngram_length, skip_size=self.skip_length)
        return Counter([skip.string for skip in skip_gen])

    def _index_term_skips(self, term: str):
        """Computes and stores the skipgram frequency vector for an already-vocabulary-indexed
        term, and marks its length bucket dirty so it gets rebuilt on next use."""
        term_id = self.vocabulary.term_id[term]
        skipgram_freq = self._term_to_skip(term)
        self.skipgram_vocabulary.add_terms(list(skipgram_freq.keys()))
        self._term_skipgram_freq[term_id] = skipgram_freq
        self._term_ids_by_length[len(term)].add(term_id)
        # mark this length as needing a bucket rebuild; lengths that weren't
        # touched by this call keep their already-built bucket as is.
        self._dirty_lengths.add(len(term))

    def _build_bucket(self, term_length: int):
        """(Re)build the sparse matrix and vector lengths for a single term
        length bucket. Only called for lengths that are dirty (new terms of
        that length were indexed since the bucket was last built) or that
        have never been built, so indexing terms of one length never forces
        recomputation of buckets for other lengths.
        """
        term_ids = np.array(sorted(self._term_ids_by_length[term_length]), dtype=np.int64)
        local_col = {int(term_id): local for local, term_id in enumerate(term_ids)}
        rows, cols, data = [], [], []
        for term_id in term_ids:
            for skipgram, freq in self._term_skipgram_freq[int(term_id)].items():
                rows.append(self.skipgram_vocabulary.term_id[skipgram])
                cols.append(local_col[int(term_id)])
                data.append(freq)
        num_skipgrams = len(self.skipgram_vocabulary)
        bucket_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(num_skipgrams, len(term_ids)), dtype=np.float64
        )
        self._length_buckets[term_length] = bucket_matrix
        self._length_bucket_term_ids[term_length] = term_ids
        bucket_vl = np.sqrt(bucket_matrix.multiply(bucket_matrix).sum(axis=0)).A1
        for term_id, vl in zip(term_ids, bucket_vl):
            self._vector_length[int(term_id)] = float(vl)
        self._dirty_lengths.discard(term_length)

    def _get_bucket(self, term_length: int):
        """Return the (matrix, term_ids) bucket for a given term length,
        (re)building it first if it is dirty or has never been built. Returns
        None if no indexed terms have that length.
        """
        if term_length not in self._term_ids_by_length:
            return None
        if term_length in self._dirty_lengths or term_length not in self._length_buckets:
            self._build_bucket(term_length)
        return self._length_buckets[term_length], self._length_bucket_term_ids[term_length]

    def _get_term_vector_length(self, term, skipgram_freq):
        """Returns the cached vector length for an indexed term, or computes it on the fly
        for an unindexed term/query."""
        if term in self.vocabulary.term_id:
            term_id = self.vocabulary.term_id[term]
            self._get_bucket(len(term))
            return self._vector_length[term_id]
        return vector_length(skipgram_freq)

    def _compute_dot_product(self, term: str) -> Dict[int, float]:
        """Compute the cosine similarity between the skipgram vector of `term`
        and the skipgram vectors of all indexed terms within `max_length_diff`
        characters of `term`'s length. Each candidate term length is handled
        with its own sparse matrix-vector multiplication, so terms outside
        the length range are never multiplied against at all (matching the
        pruning behaviour of the original per-length-bucket implementation).
        """
        skipgram_freq = self._term_to_skip(term)
        term_vl = self._get_term_vector_length(term, skipgram_freq)
        if term_vl == 0:
            return {}

        num_skipgrams = len(self.skipgram_vocabulary)
        query_cols, query_data = [], []
        for skipgram, freq in skipgram_freq.items():
            skipgram_id = self.skipgram_vocabulary.term2id(skipgram)
            if skipgram_id is None:
                continue
            query_cols.append(skipgram_id)
            query_data.append(freq)
        if not query_cols:
            return {}
        query_vector = sp.csr_matrix(
            (query_data, ([0] * len(query_cols), query_cols)), shape=(1, num_skipgrams), dtype=np.float64
        )

        dot_product: Dict[int, float] = {}
        for term_length in range(len(term) - self.max_length_diff, len(term) + self.max_length_diff + 1):
            bucket = self._get_bucket(term_length)
            if bucket is None:
                continue
            bucket_matrix, bucket_term_ids = bucket
            # a bucket built before later terms introduced new skipgrams has
            # fewer rows than the current query vector; the terms in that
            # bucket can't reference those newer skipgram ids (they didn't
            # exist yet), so slicing the query down to the bucket's row
            # count is equivalent to padding the bucket with zero rows.
            num_bucket_skipgrams = bucket_matrix.shape[0]
            bucket_dot_products = (query_vector[:, :num_bucket_skipgrams] @ bucket_matrix).toarray().ravel()
            nonzero = np.where(bucket_dot_products != 0)[0]
            for local_idx in nonzero:
                term_id = int(bucket_term_ids[local_idx])
                score = bucket_dot_products[local_idx] / (term_vl * self._vector_length[term_id])
                dot_product[term_id] = float(score)
        return dot_product

    def rank_similar(self, term: str, top_n: int = 10, score_cutoff: float = 0.5):
        """Return a ranked list of similar terms from the index for a given input term,
        based on their character skipgram cosine similarity.

        :param term: a term (any string) to match against the indexed terms
        :type term: str
        :param top_n: the number of highest ranked terms to return
        :type top_n: int (default 10)
        :param score_cutoff: the minimum similarity score after which to cutoff the ranking
        :type score_cutoff: float
        :return: a ranked list of terms and their similarity scores
        :rtype: List[Tuple[str, float]]
        """
        dot_product = self._compute_dot_product(term)
        if not dot_product:
            return []
        term_ids = np.fromiter(dot_product.keys(), dtype=np.int64)
        scores = np.fromiter(dot_product.values(), dtype=np.float64)
        # partial sort: find the top_n highest-scoring candidates, then sort just those
        n = min(top_n, term_ids.size)
        top_idx = np.argpartition(-scores, n - 1)[:n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        top_terms = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < score_cutoff:
                break
            top_terms.append((self.vocabulary.id_term[int(term_ids[idx])], score))
        return top_terms
