"""Compares word frequencies between two corpora to detect words that increase, decrease,
emerge, disappear, or get replaced/shifted to a semantically similar word (via embeddings).

``SpellingCompare`` computes, for each high-frequency word, how its relative frequency
changed between two corpora, and uses an embedding model's similarity to link a word
that dropped in frequency in one corpus to a similar word that jumped in frequency in
the other (e.g. detecting spelling variant shifts).
"""

from collections import Counter
from collections import namedtuple
from typing import List, Set, Union

from .freq import compute_llr

Diff = namedtuple('Diff', "word freq1 frac1 perc_diff1 freq2 frac2 perc_diff2 llr direction")


class Change:
    """Represents a detected change between two corpora linking a decreasing word to an
    increasing word.

    Attributes:
        decrease_word (str, optional): The word that decreased/disappeared.
        decrease_type (str, optional): The type of decrease ('decrease' or 'disappear').
        increase_word (str, optional): The word that increased/emerged.
        increase_type (str, optional): The type of increase ('increase' or 'emerge').
        sim_score (float, optional): The embedding similarity score between the two words.
    """

    def __init__(self, decrease_word: str = None, decrease_type: str = None,
                 increase_word: str = None, increase_type: str = None,
                 sim_score: float = None):
        """Initializes a Change record.

        Args:
            decrease_word (str, optional): The word that decreased/disappeared.
            decrease_type (str, optional): The type of decrease ('decrease' or 'disappear').
            increase_word (str, optional): The word that increased/emerged.
            increase_type (str, optional): The type of increase ('increase' or 'emerge').
            sim_score (float, optional): The embedding similarity score between the two words.
        """
        self.decrease_word = decrease_word
        self.decrease_type = decrease_type
        self.increase_word = increase_word
        self.increase_type = increase_type
        self.sim_score = sim_score


class SpellingCompare:
    """Compares word frequencies between two corpora and links words that dropped in
    one corpus to similar words that rose in the other, using word embeddings.

    Attributes:
        word_freq1 (Counter): Word frequencies in the first corpus.
        word_freq2 (Counter): Word frequencies in the second corpus.
        embeddings: A word embedding model exposing ``wv.similarity(word1, word2)``
            (e.g. a gensim ``Word2Vec``-like model).
        total1 (int): The total token count of the first corpus.
        total2 (int): The total token count of the second corpus.
        word_frac1 (Dict[str, float]): Relative frequency of each word in the first corpus.
        word_frac2 (Dict[str, float]): Relative frequency of each word in the second corpus.
    """

    def __init__(self, word_freq1: Counter, word_freq2: Counter, embeddings):
        """Initializes the SpellingCompare with word frequencies from two corpora.

        Args:
            word_freq1 (Counter): Word frequencies in the first corpus.
            word_freq2 (Counter): Word frequencies in the second corpus.
            embeddings: A word embedding model exposing ``wv.similarity(word1, word2)``.
        """
        self.word_freq1 = word_freq1
        self.word_freq2 = word_freq2
        self.embeddings = embeddings
        self.total1 = sum(self.word_freq1.values())
        self.total2 = sum(self.word_freq2.values())
        self.word_frac1 = {word: self.word_freq1[word] / self.total1 for word in self.word_freq1}
        self.word_frac2 = {word: self.word_freq2[word] / self.total2 for word in self.word_freq2}

    def get_frequency_change_words(self, hf_words: Union[List[str], Set[str]],
                                   increase_threshold: float = 0.5,
                                   emerge_threshold: float = 5.0,
                                   decrease_threshold: float = 0.5,
                                   disappear_threshold: float = 5.0,
                                   similarity_threshold: float = 0.6):
        """Classifies frequency changes for a set of high-frequency words and links
        decreasing/disappearing words to similar increasing/emerging words.

        Words are first sorted into 'decrease'/'disappear' (dropped from corpus 1 to 2)
        and 'increase'/'emerge' (rose from corpus 1 to 2) buckets based on the percentage
        difference thresholds (see :func:`sort_drop_jump`). Each dropped word is then
        compared via embedding similarity to every risen word; if the similarity exceeds
        ``similarity_threshold``, the pair is recorded as a 'replace' (if the word
        disappeared) or 'shift' (if it merely decreased) change. Dropped or risen words
        with no matching counterpart are recorded as unlinked changes.

        Args:
            hf_words (Union[List[str], Set[str]]): The high-frequency words to compare.
            increase_threshold (float, optional): Minimum percentage increase to count as
                'increase'. Defaults to 0.5.
            emerge_threshold (float, optional): Minimum percentage increase to count as
                'emerge'. Defaults to 5.0.
            decrease_threshold (float, optional): Minimum percentage decrease to count as
                'decrease'. Defaults to 0.5.
            disappear_threshold (float, optional): Minimum percentage decrease to count as
                'disappear'. Defaults to 5.0.
            similarity_threshold (float, optional): Minimum embedding similarity required
                to link a dropped word to a risen word. Defaults to 0.6.

        Returns:
            List[tuple]: A list of (drop_diff, jump_diff, drop_level, jump_level, change_type)
            tuples describing each detected change (drop_diff/jump_diff or their levels may
            be None when a word has no linked counterpart).
        """
        assert emerge_threshold >= increase_threshold, \
            "emerge_threshold must be equal or bigger than increase_threshold"
        assert disappear_threshold >= decrease_threshold, \
            "disappear_threshold must be equal or bigger than decrease_threshold"

        drop, jump = sort_drop_jump(hf_words, self,
                                    increase_threshold=increase_threshold,
                                    emerge_threshold=emerge_threshold,
                                    decrease_threshold=decrease_threshold,
                                    disappear_threshold=disappear_threshold)

        changes = []
        linked_jumps = set()
        for drop_level in drop:
            for drop_diff in drop[drop_level]:
                change_type = None
                for jump_level in jump:
                    for jump_diff in jump[jump_level]:
                        sim_score = self.embeddings.wv.similarity(drop_diff.word, jump_diff.word)
                        if sim_score > similarity_threshold:
                            if drop_level == 'disappear':
                                change_type = 'replace'
                            elif drop_level == 'decrease':
                                change_type = 'shift'
                            changes.append((drop_diff, jump_diff, drop_level, jump_level, change_type))
                            linked_jumps.add(jump_diff)
                if change_type is None:
                    changes.append((drop_diff, None, drop_level, None, drop_level))
        for jump_level in jump:
            for jump_diff in jump[jump_level]:
                if jump_diff not in linked_jumps:
                    changes.append((None, jump_diff, None, jump_level, jump_level))
                    linked_jumps.add(jump_diff)
        return changes

    def get_high_frequency_words(self, min_freq: int = None, min_frac: float = None):
        """Returns the union of words meeting a minimum frequency/fraction threshold in
        either corpus.

        Args:
            min_freq (int, optional): Minimum absolute frequency in either corpus.
            min_frac (float, optional): Minimum relative frequency (fraction) in either
                corpus. Only used if ``min_freq`` is not given.

        Returns:
            List[str]: The sorted list of words meeting the threshold in corpus 1 or corpus 2.

        Raises:
            ValueError: If neither ``min_freq`` nor ``min_frac`` is given.
        """
        if min_freq:
            counter1 = self.word_freq1
            counter2 = self.word_freq2
            min_value = min_freq
        elif min_frac:
            counter1 = self.word_frac1
            counter2 = self.word_frac2
            min_value = min_frac
        else:
            raise ValueError('must pass either min_freq or min_frac to select high_frequency words')
        hf_words1, hf_words2 = set(), set()
        for word, value in counter1.most_common():
            if value >= min_value:
                hf_words1.add(word)
        for word, value in counter2.most_common():
            if value >= min_value:
                hf_words2.add(word)
        hf_words_both = sorted(hf_words1.union(hf_words2))
        return hf_words_both

    def compute_percentage_diff(self, word: str):
        """Computes the frequency and percentage-difference statistics for a word between
        the two corpora.

        Args:
            word (str): The word to compute statistics for.

        Returns:
            Diff: A namedtuple with the word's frequency, fraction and percentage
            difference in each corpus, plus the log-likelihood ratio and its direction.
        """
        freq1 = self.word_freq1[word] if word in self.word_freq1 else 0
        freq2 = self.word_freq2[word] if word in self.word_freq2 else 0
        frac1 = freq1 / self.total1
        frac2 = freq2 / self.total2
        if freq1 == 0:
            perc_diff1 = (frac1 - frac2) / frac2
            perc_diff2 = 1e4
        elif freq2 == 0:
            perc_diff1 = 1e4
            perc_diff2 = (frac2 - frac1) / frac1
        else:
            perc_diff1 = (frac1 - frac2) / frac2
            perc_diff2 = (frac2 - frac1) / frac1
        llr, direction = compute_llr(word, self.word_freq1, self.total1, self.word_freq2, self.total2,
                                     include_direction=True)
        return Diff(word, freq1, frac1, perc_diff1, freq2, frac2, perc_diff2, llr, direction)


def sort_drop_jump(hf_words: Union[Set[str], List[str]],
                   spelling_compare: SpellingCompare,
                   increase_threshold: float = 0.5,
                   emerge_threshold: float = 5.0,
                   decrease_threshold: float = 0.5,
                   disappear_threshold: float = 5.0):
    """Sorts high-frequency words into 'decrease'/'disappear' and 'increase'/'emerge'
    buckets based on their percentage frequency difference between two corpora.

    Args:
        hf_words (Union[Set[str], List[str]]): The high-frequency words to classify.
        spelling_compare (SpellingCompare): Provides per-word percentage-difference stats.
        increase_threshold (float, optional): Minimum percentage increase to count as
            'increase'. Defaults to 0.5.
        emerge_threshold (float, optional): Minimum percentage increase to count as
            'emerge' (checked before 'increase'). Defaults to 5.0.
        decrease_threshold (float, optional): Minimum percentage decrease to count as
            'decrease'. Defaults to 0.5.
        disappear_threshold (float, optional): Minimum percentage decrease to count as
            'disappear' (checked before 'decrease'). Defaults to 5.0.

    Returns:
        Tuple[Dict[str, list], Dict[str, list]]: A pair of dicts (drop, jump), each
        mapping a change-level name ('decrease'/'disappear' or 'emerge'/'increase') to
        the list of Diff records at that level.
    """
    drop = {
        'decrease': [],
        'disappear': []
    }
    jump = {
        'emerge': [],
        'increase': []
    }
    for hf_word in hf_words:
        diff = spelling_compare.compute_percentage_diff(hf_word)
        if diff.perc_diff1 > emerge_threshold:
            drop['disappear'].append(diff)
        elif diff.perc_diff1 > increase_threshold:
            drop['decrease'].append(diff)
        elif diff.perc_diff2 > disappear_threshold:
            jump['emerge'].append(diff)
        elif diff.perc_diff2 > decrease_threshold:
            jump['increase'].append(diff)
    return drop, jump
