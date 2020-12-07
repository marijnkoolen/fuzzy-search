from unittest import TestCase
from fuzzy_search.fuzzy_phrase_model import PhraseModel


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
