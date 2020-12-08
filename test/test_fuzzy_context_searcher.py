from unittest import TestCase

from fuzzy_search.fuzzy_match import Match, MatchInContext
from fuzzy_search.fuzzy_context_searcher import FuzzyContextSearcher
from fuzzy_search.fuzzy_phrase import Phrase
from fuzzy_search.fuzzy_phrase_model import PhraseModel


class TestFuzzyContextSearcher(TestCase):

    def setUp(self) -> None:
        self.text = "This string contains test text."
        self.phrase = Phrase("test")
        self.searcher = FuzzyContextSearcher()
        self.match = Match(self.phrase, self.phrase, "test", 21)

    def test_fuzzy_context_searcher(self):
        error = None
        try:
            FuzzyContextSearcher()
        except any as err:
            error = err
        self.assertEqual(error, None)

    def test_fuzzy_context_searcher_can_add_context(self):
        match_in_context = self.searcher.add_match_context(self.match, self.text)
        self.assertEqual(match_in_context.context_start, 0)

    def test_fuzzy_context_searcher_can_search_context(self):
        match_in_context = self.searcher.add_match_context(self.match, self.text)
        phrase_model = PhraseModel(model=[{"phrase": "contains"}])
        context_searcher = FuzzyContextSearcher()
        context_searcher.index_phrase_model(phrase_model)
        print(match_in_context.context)
        matches = context_searcher.find_matches_in_context(match_in_context)
        self.assertEqual(len(matches), 1)
