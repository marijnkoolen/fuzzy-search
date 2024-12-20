from unittest import TestCase

from fuzzy_search.phrase.phrase_model import PhraseModel
from fuzzy_search.tokenization.token import Tokenizer


class Test(TestCase):

    def test_making_empty_phrase_model(self):
        phrase_model = PhraseModel()
        self.assertNotEqual(phrase_model, None)

    def test_making_phrase_model_with_list_of_keyword_strings(self):
        phrases = ["test"]
        phrase_model = PhraseModel(phrases=phrases)
        self.assertEqual(phrase_model.has_phrase(phrases[0]), True)

    def test_making_phrase_model_with_list_of_phrase_dictionaries(self):
        phrases = [{"phrase": "test"}]
        phrase_model = PhraseModel(phrases=phrases)
        self.assertEqual(phrase_model.has_phrase(phrases[0]["phrase"]), True)

    def test_phrase_model_can_add_phrase(self):
        phrases = [{"phrase": "test"}]
        phrase_model = PhraseModel()
        phrase_model.add_phrases(["test"])
        self.assertEqual(phrase_model.has_phrase(phrases[0]["phrase"]), True)

    def test_phrase_model_can_remove_phrase(self):
        phrases = [{"phrase": "test"}]
        phrase_model = PhraseModel(phrases=phrases)
        phrase_model.remove_phrases(["test"])
        self.assertEqual(phrase_model.has_phrase(phrases[0]["phrase"]), False)

    def test_phrase_model_indexes_phrase_words(self):
        phrases = [{"phrase": "this is a test"}]
        phrase_model = PhraseModel(phrases=phrases)
        self.assertEqual(phrases[0]["phrase"] in phrase_model.word_in_phrase["test"], True)

    def test_can_add_label_to_phrase(self):
        phrases = [{"phrase": "test", "label": "some_value"}]
        phrase_model = PhraseModel(phrases=phrases)
        phrase_model.add_custom(phrases)
        self.assertEqual(phrase_model.has_label(phrases[0]["phrase"]), True)

    def test_can_add_label_as_list_to_phrase(self):
        phrases = [{"phrase": "test", "label": ["some_value", "other_value"]}]
        phrase_model = PhraseModel(phrases=phrases)
        phrase_model.add_custom(phrases)
        label = phrase_model.get_labels(phrases[0]["phrase"])
        self.assertEqual(isinstance(label, set), True)
        self.assertEqual(len(label), 2)

    def test_can_add_custom_key_value_pairs_to_phrase(self):
        phrases = [{"phrase": "test", "some_key": "some_value"}]
        phrase_model = PhraseModel(phrases=phrases)
        phrase_model.add_custom(phrases)
        self.assertEqual(phrase_model.has_custom(phrases[0]["phrase"], "some_key"), True)

    def test_can_add_variant_phrase(self):
        phrases = [{"phrase": "okay", "variants": ["OK"]}]
        phrase_model = PhraseModel(phrases=phrases)
        phrase_model.add_variants(phrases)
        self.assertEqual(phrase_model.variant_of("OK").phrase_string, phrases[0]["phrase"])

    def test_can_add_variant_phrase_with_max_start_offset(self):
        phrase = {"phrase": "okay", "variants": ["OK"], "max_start_offset": 1}
        phrases = [phrase]
        phrase_model = PhraseModel(phrases=phrases)
        phrase_model.add_variants(phrases)
        for vi, variant_string in enumerate(phrase_model.variant_index):
            with self.subTest(vi):
                variant = phrase_model.variant_index[variant_string]
                self.assertEqual(phrase['max_start_offset'], variant.max_start_offset)

    def test_can_add_distractor_phrase_with_max_start_offset(self):
        phrase = {"phrase": "okay", "distractors": ["OK"], "max_start_offset": 1}
        phrases = [phrase]
        phrase_model = PhraseModel(phrases=phrases)
        phrase_model.add_distractors(phrases)
        for vi, distractor_string in enumerate(phrase_model.distractor_index):
            with self.subTest(vi):
                distractor = phrase_model.distractor_index[distractor_string]
                self.assertEqual(phrase['max_start_offset'], distractor.max_start_offset)

    def test_can_add_distractors(self):
        phrases = [{"phrase": "okay", "distractors": ["OK"]}]
        phrase_model = PhraseModel(phrases=phrases)
        phrase_model.add_variants(phrases)
        self.assertEqual(phrases[0]["phrase"] in phrase_model.is_distractor_of["OK"], True)

    def test_can_configure_ngram_size(self):
        phrases = [{"phrase": "test", "label": "some_value"}]
        phrase_model = PhraseModel(phrases=phrases, config={"ngram_size": 3})
        self.assertEqual(phrase_model.ngram_size, 3)
        self.assertEqual(phrase_model.phrase_index["test"].ngram_size, 3)

    def test_can_get_json_representation(self):
        phrases = [{"phrase": "okay", "variants": ["OK"]}]
        phrase_model = PhraseModel(phrases=phrases)
        phrase_model.add_variants(phrases)
        phrase_json = phrase_model.json
        self.assertEqual(phrase_json[0]['phrase'], phrases[0]['phrase'])
        self.assertEqual(phrase_json[0]['variants'][0], phrases[0]['variants'][0])


class TestPhraseModelTokenizer(TestCase):

    def setUp(self) -> None:
        self.tokenizer = Tokenizer()
        self.phrase = 'this is a test with a repetition of test'
        self.phrases = [{"phrase": self.phrase}]
        self.phrase_model = PhraseModel(phrases=self.phrases, tokenizer=self.tokenizer)

    def test_can_add_tokenizer_at_init(self):
        phrase_model = PhraseModel(phrases=self.phrases, tokenizer=self.tokenizer)
        tokens = self.tokenizer.tokenize(self.phrase)
        self.assertEqual(True, phrase_model.has_token(tokens[0]))

    def test_phrase_model_indexes_phrase_token_min_offset(self):
        tokens = self.tokenizer.tokenize(self.phrase)
        test_token = tokens[3]
        self.assertEqual(test_token.char_index, self.phrase_model.min_token_offset_in_phrase[test_token.n][self.phrase])

    def test_phrase_model_indexes_phrase_token_max_offset(self):
        tokens = self.tokenizer.tokenize(self.phrase)
        test_token = tokens[8]
        self.assertEqual(test_token.char_index, self.phrase_model.max_token_offset_in_phrase[test_token.n][self.phrase])

    def test_phrase_model_indexes_no_phrase_token_max_start_offset(self):
        tokens = self.tokenizer.tokenize(self.phrase)
        test_token = tokens[8]
        self.assertEqual(False, test_token.n in self.phrase_model.phrase_token_max_start_offset)

    def test_phrase_model_indexes_phrase_token_max_start_offset(self):
        phrases = [{"phrase": self.phrase, 'max_start_offset': 10}]
        phrase_model = PhraseModel(phrases=phrases, tokenizer=self.tokenizer)
        tokens = self.tokenizer.tokenize(self.phrase)
        test_token = tokens[8]
        self.assertEqual(True, test_token.n in phrase_model.phrase_token_max_start_offset)

    def test_phrase_model_indexes_no_phrase_token_max_end_offset(self):
        tokens = self.tokenizer.tokenize(self.phrase)
        test_token = tokens[8]
        self.assertEqual(False, test_token.n in self.phrase_model.phrase_token_max_end_offset)

    def test_phrase_model_indexes_phrase_token_max_end_offset(self):
        phrases = [{"phrase": self.phrase, 'max_end_offset': 10}]
        phrase_model = PhraseModel(phrases=phrases, tokenizer=self.tokenizer)
        tokens = self.tokenizer.tokenize(self.phrase)
        test_token = tokens[8]
        self.assertEqual(True, test_token.n in phrase_model.phrase_token_max_end_offset)
