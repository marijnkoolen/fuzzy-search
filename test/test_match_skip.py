from unittest import TestCase

from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.tokenization.string import SkipGram
from fuzzy_search.match.skip_match import SkipMatches


class TestSkipMatches(TestCase):

    def test_skip_matches_registers_match(self):
        skip_matches = SkipMatches(2, 2)
        phrase = Phrase('test')
        skipgram = SkipGram('ts', 0, len(phrase.phrase_string), 3)
        skip_matches.add_skip_match(skipgram, phrase)
        self.assertTrue(phrase in skip_matches.match_set)

