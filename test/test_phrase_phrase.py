from typing import Generator
from unittest import TestCase

from fuzzy_search.phrase.phrase import text2skipgrams, Phrase
from fuzzy_search.tokenization.token import Tokenizer


class Test(TestCase):

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

    def test_fuzzy_phrase_can_set_max_start_offset(self):
        phrase = Phrase({"phrase": "some phrase", "max_start_offset": 3})
        self.assertEqual(phrase.max_start_offset, 3)

    def test_fuzzy_phrase_can_check_if_max_start_offset(self):
        phrase = Phrase({"phrase": "some phrase", "max_start_offset": 3})
        self.assertEqual(True, phrase.has_max_start_offset())

    def test_fuzzy_phrase_can_set_max_start_end(self):
        phrase = Phrase({"phrase": "some phrase", "max_start_offset": 3})
        self.assertEqual(phrase.max_start_end, 3 + len("some_phrase"))

    def test_fuzzy_phrase_can_check_if_max_end_offset(self):
        phrase = Phrase({"phrase": "some phrase", "max_end_offset": 3})
        self.assertEqual(True, phrase.has_max_end_offset())

    def test_fuzzy_phrase_cannot_set_negative_max_start_offset(self):
        error = None
        try:
            Phrase({"phrase": "some phrase", "max_start_offset": -3})
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

    def test_fuzzy_phrase_can_set_metadata(self):
        phrase = Phrase({"phrase": "some phrase", "metadata": {"lang": "en"}})
        self.assertEqual('en', phrase.metadata['lang'])


class TestPhraseTokens(TestCase):

    def setUp(self) -> None:
        self.phrase = "some phrase"
        self.tokenizer = Tokenizer()
        self.doc = self.tokenizer.tokenize_doc(self.phrase)

    def test_phrase_can_take_tokens(self):
        phrase = Phrase(self.phrase, tokens=self.doc.tokens)
        for doc_token, phrase_token in zip(self.doc.tokens, phrase.tokens):
            with self.subTest(doc_token.char_index):
                self.assertEqual(doc_token, phrase_token)

    def test_phrase_can_take_tokenizer(self):
        phrase = Phrase(self.phrase, tokenizer=self.tokenizer)
        for doc_token, phrase_token in zip(self.doc.tokens, phrase.tokens):
            with self.subTest(doc_token.char_index):
                self.assertEqual(doc_token.char_index, phrase_token.char_index)

    def test_phrase_with_tokens_has_token_index(self):
        phrase = Phrase(self.phrase, tokenizer=self.tokenizer)
        self.assertEqual(True, 0 in phrase.token_index['some'])

    def test_phrase_with_tokens_has_token_multi_index(self):
        phrase = Phrase('multiple occurs multiple times', tokenizer=self.tokenizer)
        self.assertEqual(2, len(phrase.token_index['multiple']))
