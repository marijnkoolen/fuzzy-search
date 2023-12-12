from unittest import TestCase

from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.tokenization.string import SkipGram
from fuzzy_search.search.phrase_searcher import Candidate


class TestCandidate(TestCase):

    def test_candidate_detects_no_match_with_no_skip_match(self):
        phrase = Phrase('test')
        candidate = Candidate(phrase)
        self.assertEqual(candidate.is_match(0.5), False)

    def test_candidate_detects_no_match(self):
        phrase = Phrase('test')
        candidate = Candidate(phrase)
        skipgram = SkipGram('ts', 0, len(phrase.phrase_string), 3)
        candidate.add_skip_match(skipgram)
        self.assertEqual(candidate.is_match(0.5), False)

    def test_candidate_has_skipgram_overlap(self):
        phrase = Phrase('test')
        candidate = Candidate(phrase)
        skipgram = SkipGram('ts', 0, len(phrase.phrase_string), 3)
        candidate.add_skip_match(skipgram)
        self.assertTrue(candidate.get_skip_set_overlap() > 0.0)


