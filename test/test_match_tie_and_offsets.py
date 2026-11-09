from unittest import TestCase

from fuzzy_search.match.candidate_match import Candidate
from fuzzy_search.match.match_offsets import filter_matches_by_overlap
from fuzzy_search.match.phrase_match import PhraseMatch
from fuzzy_search.match.skip_match import filter_overlapping_phrase_candidates
from fuzzy_search.phrase.phrase import Phrase


class TestCandidateTieBreak(TestCase):

    def setUp(self) -> None:
        self.main = Phrase('abcd efgh')
        self.variant_x = Phrase('abcd efgX')
        self.variant_y = Phrase('abcd efgY')

    def test_main_phrase_wins_tie(self):
        candidates = [Candidate(self.variant_x, 0, 9, 'abcd efgZ'), Candidate(self.main, 0, 9, 'abcd efgZ')]
        filtered = filter_overlapping_phrase_candidates(candidates, main_phrase_string='abcd efgh')
        self.assertEqual(['abcd efgh'], [c.phrase.phrase_string for c in filtered])

    def test_variant_tie_is_alphabetical_regardless_of_input_order(self):
        for order in ([self.variant_y, self.variant_x], [self.variant_x, self.variant_y]):
            candidates = [Candidate(phrase, 0, 9, 'abcd efgZ') for phrase in order]
            filtered = filter_overlapping_phrase_candidates(candidates, main_phrase_string='abcd efgh')
            self.assertEqual(['abcd efgX'], [c.phrase.phrase_string for c in filtered])


class TestFirstBest(TestCase):

    def test_first_best_applies_to_every_overlap_group(self):
        first, second = Phrase('some phrase'), Phrase('some phrasf')
        matches = [PhraseMatch(phrase, phrase, 'some phrase', offset, levenshtein_similarity=0.9)
                   for offset in (0, 50) for phrase in (first, second)]
        filtered = filter_matches_by_overlap(matches, first_best=True)
        self.assertEqual([0, 50], [match.offset for match in filtered])


class TestMaxOffsetAlias(TestCase):

    def test_max_offset_sets_max_start_offset(self):
        phrase = Phrase({'phrase': 'PRAESIDE', 'max_offset': 4})
        self.assertEqual(4, phrase.max_start_offset)

    def test_max_start_offset_takes_precedence(self):
        phrase = Phrase({'phrase': 'PRAESIDE', 'max_offset': 4, 'max_start_offset': 10})
        self.assertEqual(10, phrase.max_start_offset)


class TestPhraseThresholds(TestCase):

    def setUp(self) -> None:
        from fuzzy_search.search.phrase_searcher import FuzzyPhraseSearcher
        self.searcher_class = FuzzyPhraseSearcher
        self.text = 'Ontfangen eenen brieff van den Heer'

    def matches(self, phrase_model):
        searcher = self.searcher_class(phrase_model=phrase_model, config={'levenshtein_threshold': 0.5})
        return [match.phrase.phrase_string for match in searcher.find_matches(self.text)]

    def test_phrase_threshold_overrides_searcher(self):
        self.assertEqual(['Ontfangen een Missive van'], self.matches([{'phrase': 'Ontfangen een Missive van'}]))
        strict = [{'phrase': 'Ontfangen een Missive van', 'levenshtein_threshold': 0.9}]
        self.assertEqual([], self.matches(strict))

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            Phrase({'phrase': 'De Heeren', 'levenshtein_threshold': 1.5})
        with self.assertRaises(TypeError):
            Phrase({'phrase': 'De Heeren', 'ngram_threshold': '0.5'})
