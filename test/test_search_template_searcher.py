from unittest import TestCase

from fuzzy_search.phrase.phrase_model import PhraseModel
from fuzzy_search.pattern.fuzzy_template import FuzzyTemplate
from fuzzy_search.search.template_searcher import FuzzyTemplateSearcher, get_phrase_match_list_labels
from fuzzy_search.match.phrase_match import PhraseMatch

from data.demo_data import DemoData


class TestFuzzyTemplateSearcher(TestCase):

    def setUp(self) -> None:
        self.phrase_model = PhraseModel([{"phrase": "test", "label": "test_label"}])
        self.template = FuzzyTemplate(self.phrase_model, template_json=["test_label"])

    def test_can_make_searcher(self):
        searcher = FuzzyTemplateSearcher()
        self.assertEqual(searcher.template, None)

    def test_can_add_template_at_init(self):
        searcher = FuzzyTemplateSearcher(template=self.template)
        self.assertEqual(searcher.template, self.template)

    def test_can_add_template_later(self):
        searcher = FuzzyTemplateSearcher()
        searcher.set_template(self.template)
        self.assertEqual(searcher.template, self.template)

    def test_add_template_sets_phrase_model(self):
        searcher = FuzzyTemplateSearcher()
        searcher.set_template(self.template)
        self.assertEqual(searcher.phrase_model, self.phrase_model)

    def test_configure_ngram_size(self):
        searcher = FuzzyTemplateSearcher(config={"ngram_size": 3})
        phrase_model = PhraseModel([{"phrase": "test", "label": "test_label"}], config={"ngram_size": 3})
        template = FuzzyTemplate(phrase_model, template_json=["test_label"])
        searcher.set_template(template)
        self.assertEqual(searcher.ngram_size, 3)

    def test_throws_error_for_mismatch_ngram_size(self):
        searcher = FuzzyTemplateSearcher(config={"ngram_size": 3})
        phrase_model = PhraseModel([{"phrase": "test", "label": "test_label"}], config={"ngram_size": 2})
        template = FuzzyTemplate(phrase_model, template_json=["test_label"])
        error = None
        try:
            searcher.set_template(template)
        except ValueError as err:
            error = err
        self.assertNotEqual(error, None)

    def test_can_search_text(self):
        searcher = FuzzyTemplateSearcher(template=self.template)
        template_matches = searcher.search_text({"text": "this is a test", "id": "urn:this:text"})
        self.assertEqual(len(template_matches), 1)

    def test_search_text_returns_template_matches(self):
        searcher = FuzzyTemplateSearcher(template=self.template)
        template_matches = searcher.search_text({"text": "this is a test", "id": "urn:this:text"})
        self.assertEqual(len(template_matches), 1)


class TestFuzzyTemplateSearcherWithRealData(TestCase):

    def setUp(self) -> None:
        demo_data = DemoData()
        self.auction_data = demo_data.get_dataset("auction_advertisements")
        self.phrase_model = PhraseModel(model=self.auction_data["phrases"])
        self.template = FuzzyTemplate(phrase_model=self.phrase_model, template_json=self.auction_data["template"],
                                      ignore_unknown=True)
        self.auction_tests = self.auction_data["tests"]

    def prep_test(self, test_name):
        test_data = self.auction_tests[test_name]
        matches = []
        for index, match_label in enumerate(test_data["match_sequence"]):
            match_phrase = None
            for phrase in self.phrase_model.get_phrases():
                if phrase.has_label(match_label):
                    match_phrase = phrase
                    break
            if match_phrase is None:
                raise ValueError(f"No phrase with label {match_label}")
            match = PhraseMatch(match_phrase=match_phrase, match_variant=match_phrase,
                                match_string=match_label, match_offset=index)
            match.label = match_label
            matches.append(match)
        return matches, test_data

    def test_search_text_finds_template_with_auction_test_1(self):
        test_matches, test_data = self.prep_test("test1")
        test_template_labels = test_data["template_matches"][0]
        searcher = FuzzyTemplateSearcher(template=self.template)
        template_matches = searcher.find_template_matches(test_matches)
        self.assertEqual(len(template_matches), 1)
        labels = get_phrase_match_list_labels(template_matches[0].phrase_matches)
        self.assertEqual(len(labels), len(test_template_labels))
        is_same = True
        for li, label in enumerate(labels):
            if label != test_template_labels[li]:
                is_same = False
                break
        self.assertEqual(is_same, True)

    def test_search_text_finds_template_with_auction_test_2(self):
        test_matches, test_data = self.prep_test("test2")
        searcher = FuzzyTemplateSearcher(template=self.template)
        template_matches = searcher.find_template_matches(test_matches)
        self.assertEqual(len(template_matches), test_data["num_template_matches"])

    def test_search_text_finds_template_with_auction_test_3(self):
        test_matches, test_data = self.prep_test("test3")
        searcher = FuzzyTemplateSearcher(template=self.template)
        template_matches = searcher.find_template_matches(test_matches)
        self.assertEqual(len(template_matches), test_data["num_template_matches"])

    def test_search_text_finds_template_with_auction_test_4(self):
        test_matches, test_data = self.prep_test("test4")
        searcher = FuzzyTemplateSearcher(template=self.template)
        template_matches = searcher.find_template_matches(test_matches)
        self.assertEqual(len(template_matches), test_data["num_template_matches"])
