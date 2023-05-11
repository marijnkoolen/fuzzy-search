from unittest import TestCase

from fuzzy_search.pattern.fuzzy_template import FuzzyTemplate, FuzzyTemplateGroupElement, FuzzyTemplateLabelElement
from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.phrase.phrase_model import PhraseModel
from data.demo_data import DemoData


class TestFuzzyTemplateElement(TestCase):

    def test_template_accepts_label_only(self):
        template = FuzzyTemplateLabelElement(label="some_label")
        self.assertEqual(template.label, "some_label")

    def test_template_accepts_label_and_cardinality(self):
        template = FuzzyTemplateLabelElement(label="some_label", cardinality="multi")
        self.assertEqual(template.cardinality, "multi")

    def test_template_rejects_invalid_cardinality_value(self):
        error = None
        try:
            FuzzyTemplateLabelElement(label="some_label", cardinality="more_than_one")
        except ValueError as err:
            error = err
        self.assertNotEqual(error, None)


class TestFuzzyTemplateGroup(TestCase):

    def setUp(self) -> None:
        self.element = FuzzyTemplateLabelElement(label="element_label")

    def test_template_group_accepts_label_only(self):
        template_group = FuzzyTemplateGroupElement([self.element], label="group_label")
        self.assertEqual(template_group.label, "group_label")

    def test_template_group_accepts_label_and_order(self):
        template_group = FuzzyTemplateGroupElement([self.element], label="group_label")
        self.assertEqual(template_group.ordered, True)


class TestFuzzyTemplate(TestCase):

    def setUp(self) -> None:
        phrase_model = [{"phrase": "test phrase", "label": ["test_label"]}]
        self.phrase_model = PhraseModel(model=phrase_model)

    def test_template_generation(self):
        template = FuzzyTemplate(phrase_model=self.phrase_model, template_json={"label": "test_label"})
        self.assertEqual(isinstance(template, FuzzyTemplate), True)

    def test_template_can_get_phrase_by_label(self):
        template = FuzzyTemplate(phrase_model=self.phrase_model, template_json={"label": "test_label"})
        phrases = template.get_label_phrases("test_label")
        self.assertEqual(len(phrases), 1)

    def test_template_get_phrase_by_label_returns_phrase_object(self):
        template = FuzzyTemplate(phrase_model=self.phrase_model, template_json={"label": "test_label"})
        phrases = template.get_label_phrases("test_label")
        self.assertEqual(isinstance(phrases[0], Phrase), True)

    def test_template_get_phrase_by_label_returns_correct_phrase(self):
        template = FuzzyTemplate(phrase_model=self.phrase_model, template_json={"label": "test_label"})
        phrases = template.get_label_phrases("test_label")
        self.assertEqual(phrases[0].phrase_string, "test phrase")

    def test_template_register_simple_element(self):
        template = FuzzyTemplate(phrase_model=self.phrase_model,
                                 template_json={"label": "test_label", "extra": "extra"})
        self.assertEqual(template.has_label("test_label"), True)

    def test_template_register_simple_element_as_multi_if_no_cardinality(self):
        template = FuzzyTemplate(phrase_model=self.phrase_model, template_json=[{"label": "test_label"}])
        self.assertEqual(template.label_element_index["test_label"].cardinality, "single")

    def test_template_register_simple_element_with_list_labels(self):
        template = FuzzyTemplate(phrase_model=self.phrase_model,
                                 template_json=[{"label": "test_label", "required": True}])
        element = template.get_element("test_label")
        self.assertEqual(element.required, True)

    def test_template_cannot_register_element_with_unknown_label(self):
        error = None
        try:
            FuzzyTemplate(phrase_model=self.phrase_model, template_json=[{"label": "other_label", "required": True}])
        except ValueError as err:
            error = err
        self.assertNotEqual(error, None)

    def test_template_can_ignore_element_with_unknown_label(self):
        template = FuzzyTemplate(phrase_model=self.phrase_model,
                                 template_json=[{"label": "other_label", "required": True}],
                                 ignore_unknown=True)
        self.assertEqual(template.has_label("other_label"), False)

    def test_template_can_return_required_elements(self):
        template = FuzzyTemplate(phrase_model=self.phrase_model, template_json=[{"label": "test_label"}])
        self.assertEqual(len(template.get_required_elements()), 0)

    def test_template_returns_all_required_elements(self):
        template = FuzzyTemplate(phrase_model=self.phrase_model,
                                 template_json=[{"label": "test_label", "required": True}])
        self.assertEqual(len(template.get_required_elements()), 1)

    def test_template_returns_all_required_element_labels(self):
        template = FuzzyTemplate(phrase_model=self.phrase_model,
                                 template_json=[{"label": "test_label", "required": True}])
        self.assertEqual(len(template.get_required_labels()), 1)
        self.assertEqual("test_label" in template.get_required_labels(), True)

    def test_template_can_register_group_elements(self):
        template = FuzzyTemplate(phrase_model=self.phrase_model,
                                 template_json=[{"type": "group", "label": "group_label", "elements": ["test_label"]}])
        self.assertEqual(template.has_group("group_label"), True)
        self.assertEqual("group_label" in template.group_element_index, True)


class TestFuzzyTemplateWithRealData(TestCase):

    def setUp(self) -> None:
        demo_data = DemoData()
        self.auction_data = demo_data.get_dataset("auction_advertisements")

    def test_template_can_read_real_data(self):
        phrase_model = PhraseModel(model=self.auction_data["phrases"])
        template = FuzzyTemplate(phrase_model=phrase_model,
                                 template_json=self.auction_data["template"],
                                 ignore_unknown=True)
        self.assertEqual(template.has_label("product_unit"), True)
