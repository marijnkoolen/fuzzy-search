from unittest import TestCase
from typing import Generator
from fuzzy_search.fuzzy_phrase import text2skipgrams


class Test(TestCase):

    # text2skipgrams

    def test_text2skipgrams_rejects_negative_ngram_size(self):
        error = None
        try:
            next(text2skipgrams("test", ngram_size=-1, skip_size=2))
        except ValueError as err:
            error = err
        self.assertNotEqual(error, None)

    def test_text2skipgrams_rejects_negative_skip_size(self):
        error = None
        try:
            next(text2skipgrams("test", ngram_size=2, skip_size=-1))
        except ValueError as err:
            error = err
        self.assertNotEqual(error, None)

    def test_text2skipgrams_accepts_positive_ngram_size(self):
        skip_iter = text2skipgrams("test", ngram_size=2, skip_size=2)
        self.assertEqual(isinstance(skip_iter, Generator), True)

    def test_skipgrams_have_correct_length(self):
        skipgrams = [skipgram for skipgram in text2skipgrams("test", ngram_size=2, skip_size=2)]
        self.assertEqual(skipgrams[2].length, 4)
