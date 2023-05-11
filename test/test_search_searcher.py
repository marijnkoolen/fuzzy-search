from unittest import TestCase

from fuzzy_search.search.config import default_config
from fuzzy_search.search.searcher import FuzzySearcher
from fuzzy_search.tokenization.token import Tokenizer


class TestSearcher(TestCase):

    def test_searcher_has_default_config(self):
        searcher = FuzzySearcher()
        for field in default_config:
            with self.subTest(field):
                self.assertEqual(default_config[field], searcher.config[field])

    def test_searcher_has_default_tokenizer(self):
        searcher = FuzzySearcher()
        self.assertEqual(True, isinstance(searcher.tokenizer, Tokenizer))

    def test_searcher_indexes_phrases(self):
        phrase = 'test'
        searcher = FuzzySearcher(phrase_list=[phrase])
        self.assertEqual(1, len(searcher.phrases))

    def test_searcher_indexes_phrase_skipgrams(self):
        phrase = 'test'
        searcher = FuzzySearcher(phrase_list=[phrase])
        self.assertEqual(True, 'te' in searcher.skipgram_index)

    def test_searcher_indexes_phrase_skipgram_phrase(self):
        phrase = 'test'
        searcher = FuzzySearcher(phrase_list=[phrase])
        self.assertEqual(True, phrase in [phrase.phrase_string for phrase in searcher.skipgram_index['te']])

    def test_searcher_passes_tokenizer_to_phrase_model(self):
        phrase = 'test'
        searcher = FuzzySearcher(phrase_list=[phrase])
        self.assertEqual(searcher.tokenizer, searcher.phrase_model.tokenizer)

    def test_searcher_indexes_phrase_tokens(self):
        phrase = 'test'
        searcher = FuzzySearcher(phrase_list=[phrase])
        self.assertEqual(True, phrase in searcher.phrase_model.token_in_phrase)
