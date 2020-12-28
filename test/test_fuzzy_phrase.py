from unittest import TestCase
from typing import Generator
from fuzzy_search.fuzzy_phrase import text2skipgrams, Phrase


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


class TestFuzzyPhrase(TestCase):

    def test_fuzzy_phrase_accepts_phrase_as_string(self):
        phrase = Phrase("some phrase")
        self.assertEqual(phrase.phrase_string, "some phrase")

    def test_fuzzy_phrase_accepts_phrase_as_dict(self):
        phrase = Phrase({"phrase": "some phrase"})
        self.assertEqual(phrase.phrase_string, "some phrase")

    def test_fuzzy_phrase_can_set_max_offset(self):
        phrase = Phrase({"phrase": "some phrase", "max_offset": 3})
        self.assertEqual(phrase.max_offset, 3)

    def test_fuzzy_phrase_can_set_max_end(self):
        phrase = Phrase({"phrase": "some phrase", "max_offset": 3})
        self.assertEqual(phrase.max_end, 3 + len("some_phrase"))

    def test_fuzzy_phrase_cannot_set_negative_max_offset(self):
        error = None
        try:
            Phrase({"phrase": "some phrase", "max_offset": -3})
        except ValueError as err:
            error = err
        self.assertNotEqual(error, None)

    def test_fuzzy_phrase_accepts_phrase_with_valid_string_label(self):
        phrase = Phrase({"phrase": "some phrase", "label": "some_label"})
        self.assertEqual(phrase.label, "some_label")

    def test_fuzzy_phrase_accepts_phrase_with_valid_list_of_strings_label(self):
        phrase = Phrase({"phrase": "some phrase", "label": ["some_label", "other_label"]})
        self.assertEqual(isinstance(phrase.label, list), True)
        self.assertEqual("other_label" in phrase.label, True)

    def test_fuzzy_phrase_rejects_phrase_with_invalid_label(self):
        error = None
        try:
            Phrase({"phrase": "some phrase", "label": {"complex": "label"}})
        except ValueError as err:
            error = err
        self.assertNotEqual(error, None)
