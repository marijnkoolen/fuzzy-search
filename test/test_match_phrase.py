from unittest import TestCase
import string
from fuzzy_search.match.phrase_match import PhraseMatch, PhraseMatchInContext, adjust_match_offsets
from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.match.phrase_match import adjust_match_start_offset, adjust_match_end_offset
from fuzzy_search.match.phrase_match import map_string


class TestFuzzyMatch(TestCase):

    def test_map_string_maps_word_string(self):
        affix_string = "test"
        mapped_string = map_string(affix_string, string.punctuation)
        self.assertEqual(mapped_string, "wwww")

    def test_map_string_maps_space_string(self):
        affix_string = "     "
        mapped_string = map_string(affix_string, string.punctuation)
        self.assertEqual(mapped_string, "sssss")

    def test_map_string_maps_mixed_string(self):
        affix_string = "a test"
        mapped_string = map_string(affix_string, string.punctuation)
        self.assertEqual(mapped_string, "wswwww")

    def test_adjust_start(self):
        text = {"text": "this text contains some words"}
        match_string = "contains"
        start = text["text"].index(match_string)
        adjusted_start = adjust_match_start_offset(text, match_string, start)
        self.assertEqual(adjusted_start, start)

    def test_adjust_start_returns_when_in_middle_of_word(self):
        text = {"text": "this text concontains some words"}
        match_string = "contains"
        start = text["text"].index(match_string)
        adjusted_start = adjust_match_start_offset(text, match_string, start)
        self.assertEqual(adjusted_start, None)

    def test_adjust_start_shifts_to_previous_character(self):
        text = {"text": "this text contains some words"}
        match_string = "ontains"
        start = text["text"].index(match_string)
        adjusted_start = adjust_match_start_offset(text, match_string, start)
        self.assertEqual(adjusted_start, start-1)

    def test_adjust_start_shifts_to_second_previous_characters(self):
        text = {"text": "this text contains some words"}
        match_string = "ntains"
        start = text["text"].index(match_string)
        adjusted_start = adjust_match_start_offset(text, match_string, start)
        self.assertEqual(adjusted_start, start-2)

    def test_adjust_start_does_not_shift_to_third_previous_character(self):
        text = {"text": "this text contains some words"}
        match_string = "tains"
        start = text["text"].index(match_string)
        adjusted_start = adjust_match_start_offset(text, match_string, start)
        self.assertEqual(adjusted_start, None)

    def test_adjust_start_shifts_to_next_character(self):
        text = {"text": "this text contains some words"}
        match_string = " contains"
        start = text["text"].index(match_string)
        adjusted_start = adjust_match_start_offset(text, match_string, start)
        self.assertEqual(adjusted_start, start+1)

    def test_adjust_end(self):
        text = {"text": "this text contains some words"}
        phrase_string = "contains"
        match_string = "contains"
        end = text["text"].index(match_string) + len(match_string)
        adjusted_end = adjust_match_end_offset(phrase_string, match_string, text, end, string.punctuation)
        self.assertEqual(adjusted_end, end)

    def test_adjust_end_shifts_back_one_when_ending_with_whitespace(self):
        text = {"text": "this text contains some words"}
        phrase_string = "contains"
        match_string = "contains "
        end = text["text"].index(match_string) + len(match_string)
        adjusted_end = adjust_match_end_offset(phrase_string, match_string, text, end, string.punctuation)
        self.assertEqual(adjusted_end, end-1)

    def test_adjust_end_shifts_back_two_when_ending_with_whitespace_and_char(self):
        text = {"text": "this text contains some words"}
        phrase_string = "contains"
        match_string = "contains s"
        end = text["text"].index(match_string) + len(match_string)
        adjusted_end = adjust_match_end_offset(phrase_string, match_string, text, end, string.punctuation)
        self.assertEqual(adjusted_end, end-2)

    def test_adjust_end_shifts_back_one_when_phrase_ends_with_whitespace(self):
        text = {"text": "this text contains some words"}
        phrase_string = "contains "
        match_string = "contains s"
        end = text["text"].index(match_string) + len(match_string)
        adjusted_end = adjust_match_end_offset(phrase_string, match_string, text, end, string.punctuation)
        self.assertEqual(adjusted_end, end-1)

    def test_adjust_end_does_not_shift_when_end_middle_of_next_word(self):
        text = {"text": "this text contains some words"}
        phrase_string = "contains"
        match_string = "contains so"
        end = text["text"].index(match_string) + len(match_string)
        adjusted_end = adjust_match_end_offset(phrase_string, match_string, text, end, string.punctuation)
        self.assertEqual(adjusted_end, None)

    def test_adjust_end_shifts_to_end_of_next_word(self):
        text = {"text": "this text contains some words"}
        phrase_string = "contains som"
        match_string = "contains som"
        end = text["text"].index(match_string) + len(match_string)
        adjusted_end = adjust_match_end_offset(phrase_string, match_string, text, end, string.punctuation)
        self.assertEqual(adjusted_end, end+1)

    def test_adjust_boundaries_removes_surrounding_whitespace(self):
        text = {"text": "this text contains some words"}
        phrase_string = "contains"
        candidate_string = " contains "
        start = text["text"].index(candidate_string)
        end = text["text"].index(candidate_string) + len(candidate_string)
        adjusted_match = adjust_match_offsets(phrase_string, candidate_string, text, start, end)
        self.assertEqual(adjusted_match["match_string"], phrase_string)

    def test_adjust_boundaries_finds_word_boundary(self):
        text = {"text": "this text contains some words"}
        phrase_string = "contains som"
        candidate_string = " contains som"
        start = text["text"].index(candidate_string)
        end = text["text"].index(candidate_string) + len(candidate_string)
        adjusted_match = adjust_match_offsets(phrase_string, candidate_string, text, start, end)
        self.assertNotEqual(adjusted_match, None)
        self.assertEqual(adjusted_match["match_string"], phrase_string + "e")


class TestMatchInContext(TestCase):

    def setUp(self) -> None:
        self.text = "This string contains test text."
        self.phrase = Phrase("test")
        self.match = PhraseMatch(self.phrase, self.phrase, "test", 21)

    def test_make_match_in_context(self):
        match_in_context = PhraseMatchInContext(self.match, self.text)
        self.assertEqual(match_in_context.context_start, 1)

    def test_context_is_adjustable(self):
        match_in_context = PhraseMatchInContext(self.match, self.text, prefix_size=10)
        self.assertEqual(match_in_context.context_start, 11)

    def test_context_contains_text_from_doc(self):
        match_in_context = PhraseMatchInContext(self.match, self.text, prefix_size=30)
        self.assertEqual(match_in_context.context, self.text)


class TestMatchAnnotation(TestCase):

    def setUp(self) -> None:
        self.text = "This string contains test text."
        self.phrase = Phrase("test")
        self.match = PhraseMatch(self.phrase, self.phrase, match_string="test", match_offset=21,
                                 text_id='test_id', match_label='test_label')

    def test_match_as_web_anno_has_multiple_bodies(self):
        match_anno = self.match.as_web_anno()
        self.assertEqual(len(match_anno['body']), 3)

    def test_match_as_web_anno_skips_correction_with_exact_match(self):
        match_anno = self.match.as_web_anno()
        label_body = None
        for body in match_anno['body']:
            if body['purpose'] == 'correcting':
                label_body = body
        self.assertEqual(label_body, None)

    def test_match_as_web_anno_has_label(self):
        match_anno = self.match.as_web_anno()
        label_body = None
        for body in match_anno['body']:
            if body['purpose'] == 'classifying':
                label_body = body
        self.assertNotEqual(label_body, None)
        self.assertEqual(label_body['value'], self.match.label)

    def test_match_in_context_as_web_anno_has_multiple_selectors(self):
        match_in_context = PhraseMatchInContext(self.match, self.text, prefix_size=10)
        match_anno = match_in_context.as_web_anno()
        self.assertEqual(isinstance(match_anno['target']['selector'], list), True)
        self.assertEqual(len(match_anno['target']['selector']), 2)

    def test_match_in_context_as_web_anno_has_text_quote_selector(self):
        match_in_context = PhraseMatchInContext(self.match, self.text, prefix_size=10)
        match_anno = match_in_context.as_web_anno()
        quote_selector = None
        for selector in match_anno['target']['selector']:
            if selector['type'] == 'TextQuoteSelector':
                quote_selector = selector
        self.assertNotEqual(quote_selector, None)
        self.assertEqual(quote_selector['prefix'], self.text[self.match.offset-10:self.match.offset])
