import math
from collections import defaultdict
from collections import Counter
from typing import Dict, Generator, Iterable, List, Tuple, Union


from fuzzy_search.tokenization.string import score_levenshtein_distance
from fuzzy_search.tokenization.string import text2skipgrams
from fuzzy_search.tokenization.vocabulary import Vocabulary


def vector_length(skipgram_freq):
    return math.sqrt(sum([skipgram_freq[skip] ** 2 for skip in skipgram_freq]))


def get_skip_coocs(seq_ids: List[str], skip_size: int = 0) -> Generator[Tuple[int, int], None, None]:
    for ci, curr_id in enumerate(seq_ids):
        for next_id in seq_ids[ci + 1: ci + 2 + skip_size]:
            yield curr_id, next_id


def get_min_length(phrase1: str, phrase2: str, begin_length: int) -> int:
    return min([len(phrase1), len(phrase2), begin_length])


def get_begin_sim(phrase1: str, phrase2: str, begin_length: int) -> float:
    if min(len(phrase1), len(phrase2)) < begin_length:
        begin_length = min(len(phrase1), len(phrase2))
    begin1 = phrase1[:begin_length]
    begin2 = phrase2[:begin_length]
    return 1 - (score_levenshtein_distance(begin1, begin2) / begin_length)


def get_end_sim(phrase1: str, phrase2: str, end_length: int) -> float:
    if min(len(phrase1), len(phrase2)) < end_length:
        end_length = min(len(phrase1), len(phrase2))
    end1 = phrase1[-end_length:]
    end2 = phrase2[-end_length:]
    return 1 - (score_levenshtein_distance(end1, end2) / end_length)


class SkipCooccurrence:

    def __init__(self, vocabulary: Vocabulary, skip_size: int = 1, sentences: Iterable[List[str]] = None):
        """A class to count the co-occurrence frequency of word skipgrams."""
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
        id1, id2 = cooc_ids
        return self.vocabulary.id2term(id1), self.vocabulary.id2term(id2)

    def get_term_coocs(self, term: str) -> Union[None, Generator[Tuple[str, str], None, None]]:
        term_id = self.vocabulary.term2id(term)
        if term_id is None:
            return None
        for cooc_ids in self.cooc_freq:
            if term_id in cooc_ids:
                yield self._cooc_ids2terms(cooc_ids), self.cooc_freq[cooc_ids]


def is_close_distance_keyword_pair(keyword1: str, keyword2: str, max_distance_ratio: float,
                                   max_length_difference: int, max_distance: int) -> bool:
    if abs(len(keyword1) - len(keyword2)) > max_length_difference:
        return False
    distance = score_levenshtein_distance(keyword1, keyword2)
    if distance < max_distance and (
            distance / len(keyword1) < max_distance_ratio or distance / len(keyword2) < max_distance_ratio):
        return True
    return False


class KeywordList:

    def __init__(self, keywords: List[str], max_length_diff: int):
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
        :
        """
        self.ngram_length = ngram_length
        self.skip_length = skip_length
        self.vocabulary = Vocabulary()
        self.vector_length = {}
        self.max_length_diff = max_length_diff
        self.skipgram_index = defaultdict(lambda: defaultdict(Counter))
        if terms is not None:
            self.index_terms(terms)

    def _reset_index(self):
        self.vocabulary.reset_index()
        self.vector_length = {}
        self.skipgram_index = defaultdict(lambda: defaultdict(Counter))

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
        skip_gen = text2skipgrams(term, ngram_size=self.ngram_length, skip_size=self.skip_length)
        return Counter([skip.string for skip in skip_gen])

    def _index_term_skips(self, term: str):
        term_id = self.vocabulary.term_id[term]
        skipgram_freq = self._term_to_skip(term)
        self.vector_length[term_id] = vector_length(skipgram_freq)
        for skipgram in skipgram_freq:
            # print(skip.string)
            self.skipgram_index[skipgram][len(term)][term_id] = skipgram_freq[skipgram]

    def _get_term_vector_length(self, term, skipgram_freq):
        if term not in self.vocabulary.term_id:
            return vector_length(skipgram_freq)
        else:
            term_id = self.vocabulary.term_id[term]
            return self.vector_length[term_id]

    def _compute_dot_product(self, term):
        skipgram_freq = self._term_to_skip(term)
        term_vl = self._get_term_vector_length(term, skipgram_freq)
        # print(term, 'vl:', term_vl)
        dot_product = defaultdict(int)
        for skipgram in skipgram_freq:
            for term_length in range(len(term) - self.max_length_diff, len(term) + self.max_length_diff + 1):
                for term_id in self.skipgram_index[skipgram][term_length]:
                    dot_product[term_id] += skipgram_freq[skipgram] * self.skipgram_index[skipgram][term_length][
                        term_id]
                    # print(term_id, self.vocab_map[term_id], dot_product[term_id])
        for term_id in dot_product:
            dot_product[term_id] = dot_product[term_id] / (term_vl * self.vector_length[term_id])
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
        top_terms = []
        for term_id in sorted(dot_product, key=lambda t: dot_product[t], reverse=True):
            if dot_product[term_id] < score_cutoff:
                break
            term = self.vocabulary.id_term[term_id]
            top_terms.append((term, dot_product[term_id]))
            if len(top_terms) == top_n:
                break
        return top_terms
