from unittest import TestCase

import fuzzy_search.match.candidate_match as can_match
from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.tokenization.string import SkipGram
from fuzzy_search.match.candidate_match import CandidatePartial


class TestCandidate(TestCase):

    def test_candidate_detects_no_match_with_no_skip_match(self):
        phrase = Phrase('test')
        candidate = CandidatePartial(phrase)
        self.assertEqual(can_match.is_match(candidate, 0.5), False)

    def test_candidate_detects_no_match(self):
        phrase = Phrase('test')
        candidate = CandidatePartial(phrase)
        skipgram = SkipGram('ts', 0, len(phrase.phrase_string), 3)
        can_match.add_skip_match(candidate, skipgram)
        self.assertEqual(can_match.is_match(candidate, 0.5), False)

    def test_candidate_has_skipgram_overlap(self):
        phrase = Phrase('test')
        candidate = CandidatePartial(phrase)
        skipgram = SkipGram('ts', 0, len(phrase.phrase_string), 3)
        can_match.add_skip_match(candidate, skipgram)
        self.assertTrue(can_match.get_skip_set_overlap(candidate) > 0.0)


