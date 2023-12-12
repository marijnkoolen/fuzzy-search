import unittest
from unittest import TestCase

from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.search.token_searcher import FuzzyTokenSearcher
from fuzzy_search.search.token_searcher import MatchType
from fuzzy_search.search.token_searcher import PartialPhraseMatch
from fuzzy_search.search.token_searcher import get_token_skipgram_matches
from fuzzy_search.search.token_searcher import get_token_skip_match_type
from fuzzy_search.search.token_searcher import get_partial_phrases
from fuzzy_search.tokenization.string import text2skipgrams
from fuzzy_search.tokenization.token import Tokenizer
from fuzzy_search.tokenization.vocabulary import Vocabulary


class TestTokenSearcher(TestCase):

    def test_token_searcher_has_default_tokenizer(self):
        token_searcher = FuzzyTokenSearcher()
        self.assertEqual(True, isinstance(token_searcher.tokenizer, Tokenizer))

    def test_token_searcher_has_token_skipgram_index(self):
        phrase_list = ['test']
        token_searcher = FuzzyTokenSearcher(phrase_list=phrase_list)
        self.assertEqual(True, phrase_list[0] in token_searcher.token_skipgram_index['te'])


def get_token_match_types(text_token, token_searcher):
    text_token_skips = [skipgram for skipgram in text2skipgrams(text_token.normalised_string,
                                                                token_searcher.ngram_size,
                                                                token_searcher.skip_size)]
    text_token_num_skips = len(text_token_skips)
    token_skip_matches = get_token_skipgram_matches(text_token_skips, token_searcher)
    match_types = []
    for match in token_skip_matches.match_start_offsets:
        print('text_token:', text_token, '\tmatch:', match)
        match_type = get_token_skip_match_type(text_token.normalised_string, text_token_num_skips,
                                               token_skip_matches, match, token_searcher, debug=2)
        match_types.append(match_type)
    return match_types


class TestTokenSearcherMatchType(TestCase):

    def setUp(self) -> None:
        phrase_list = ['test']
        self.text = 'This is a test in which we are testing'
        self.tokenizer = Tokenizer()
        self.doc = self.tokenizer.tokenize(doc_text=self.text)
        self.token_searcher = FuzzyTokenSearcher(phrase_list=phrase_list, tokenizer=self.tokenizer)

    def test_get_match_type_finds_full_match(self):
        text_token = self.doc.tokens[3]
        match_types = get_token_match_types(text_token, self.token_searcher)
        self.assertEqual(True, MatchType.FULL in match_types)

    def test_get_match_type_finds_partial_phrase_token_match(self):
        text_token = self.doc.tokens[3]
        match_types = get_token_match_types(text_token, self.token_searcher)
        self.assertEqual(True, MatchType.FULL in match_types)

    def test_get_match_type_finds_partial_text_token_match(self):
        text_token = self.doc.tokens[8]
        match_types = get_token_match_types(text_token, self.token_searcher)
        self.assertEqual(True, MatchType.PARTIAL_OF_TEXT_TOKEN in match_types)

    def test_token_searcher_finds_multiple_matches(self):
        token_matches = self.token_searcher.find_skipgram_token_matches_in_text(self.doc)
        self.assertEqual(2, len(token_matches))

    def test_token_searcher_finds_partial_phrase_token_match(self):
        self.token_searcher = FuzzyTokenSearcher(phrase_list=['test'], tokenizer=self.tokenizer)
        self.doc = self.tokenizer.tokenize(doc_text='The purpose is testing')
        token_matches = self.token_searcher.find_skipgram_token_matches_in_text(self.doc, debug=3)
        self.assertEqual(MatchType.PARTIAL_OF_TEXT_TOKEN, token_matches[0].match_type)

    def test_get_match_type_finds_multi_text_token_match(self):
        self.token_searcher = FuzzyTokenSearcher(phrase_list=['testing'], tokenizer=self.tokenizer)
        self.doc = self.tokenizer.tokenize(doc_text='We are test ing')
        token_matches = self.token_searcher.find_skipgram_token_matches_in_text(self.doc)
        self.assertEqual((self.doc.tokens[2], self.doc.tokens[3]), token_matches[0].text_tokens)

    def test_get_match_type_finds_same_text_token_match_for_multi_phrase_tokens(self):
        self.token_searcher = FuzzyTokenSearcher(phrase_list=['test', 'case'], tokenizer=self.tokenizer)
        self.doc = self.tokenizer.tokenize(doc_text='This is a testcase')
        token_matches = self.token_searcher.find_skipgram_token_matches_in_text(self.doc)
        for ti, token_match in enumerate(token_matches):
            with self.subTest(ti):
                self.assertEqual(MatchType.PARTIAL_OF_TEXT_TOKEN, token_match.match_type)


class TestPartialPhraseMatch(TestCase):

    def setUp(self) -> None:
        self.tokenizer = Tokenizer()
        self.token_searcher = FuzzyTokenSearcher(phrase_list=['test case'], tokenizer=self.tokenizer)
        self.text = 'This is a test case'
        self.phrase = Phrase(self.text, tokenizer=self.tokenizer)

    def test_making_partial_match_extends_text_tokens(self):
        doc = self.tokenizer.tokenize(doc_text=self.text)
        token_matches = self.token_searcher.find_skipgram_token_matches_in_text(doc)
        partial_match = PartialPhraseMatch(self.phrase, token_matches)
        self.assertEqual(len(partial_match.text_tokens), sum(len(tm.text_tokens) for tm in token_matches))

    def test_making_partial_match_extends_phrase_tokens(self):
        doc = self.tokenizer.tokenize(doc_text='This is a test case')
        token_matches = self.token_searcher.find_skipgram_token_matches_in_text(doc)
        partial_match = PartialPhraseMatch(self.phrase, token_matches)
        self.assertEqual(len(partial_match.phrase_tokens), sum(len(tm.phrase_tokens) for tm in token_matches))

    def test_making_partial_match_extends_text_length(self):
        doc = self.tokenizer.tokenize(doc_text='This is a test case')
        token_matches = self.token_searcher.find_skipgram_token_matches_in_text(doc)
        partial_match = PartialPhraseMatch(self.phrase, token_matches)
        text_length = token_matches[-1].text_end - token_matches[0].text_start
        self.assertEqual(partial_match.text_length, text_length)

    def test_making_partial_match_can_pop(self):
        doc = self.tokenizer.tokenize(doc_text='This is a test case')
        token_matches = self.token_searcher.find_skipgram_token_matches_in_text(doc)
        partial_match = PartialPhraseMatch(self.phrase, token_matches)
        partial_match.pop()
        self.assertEqual(len(partial_match.phrase_tokens), sum(len(tm.phrase_tokens) for tm in token_matches) - 1)

    def test_making_partial_match_picks_best_phrase_token_option(self):
        phrase = Phrase('This is a best test case')
        self.token_searcher = FuzzyTokenSearcher(phrase_list=['best test case'], tokenizer=self.tokenizer)
        doc = self.tokenizer.tokenize(doc_text='This is a best test case')
        token_matches = self.token_searcher.find_skipgram_token_matches_in_text(doc)
        for tm in token_matches:
            print('token_match:', tm)
        # partial_match = PartialPhraseMatch(token_matches, phrase=phrase)
        candidates = get_partial_phrases(token_matches, self.token_searcher)
        self.assertEqual(1, len(candidates))
        for pi, phrase in enumerate(candidates):
            print('test_search_token_searcher - phrase:', phrase)
            with self.subTest(pi):
                for partial in candidates[phrase]:
                    print('\ntest_search_token_searcher - partial:', partial)
                    self.assertEqual(0, len(partial.missing_tokens))
                self.assertEqual(2, len(candidates[phrase]))

    def test_making_partial_match_does_not_duplicate_text_tokens(self):
        phrase = Phrase('This is a best testcase', tokenizer=self.tokenizer)
        doc = self.tokenizer.tokenize(doc_text='This is a best testcase')
        token_matches = self.token_searcher.find_skipgram_token_matches_in_text(doc, debug=0)
        print('phrase:', phrase)
        for tm in token_matches:
            print('\ttoken_match:', tm)
        partial_match = PartialPhraseMatch(phrase, token_matches)
        print('partial_match.text_tokens:', partial_match.text_tokens)
        self.assertEqual(2, len(partial_match.text_tokens))

    def test_making_partial_match_discards_when_missing_phrase_token(self):
        text = 'This is a best test sentence'
        phrase = Phrase(text, tokenizer=self.tokenizer)
        self.token_searcher = FuzzyTokenSearcher(phrase_list=['test case'], tokenizer=self.tokenizer)
        doc = self.tokenizer.tokenize(doc_text=text)
        token_matches = self.token_searcher.find_skipgram_token_matches_in_text(doc, debug=3)
        for tm in token_matches:
            print(tm)
        partial_match = PartialPhraseMatch(phrase, token_matches)
        self.assertEqual(3, len(partial_match.text_tokens))

    def test_making_candidate_phrases_can_detect_term_swap(self):
        phrase = Phrase('best test case', tokenizer=self.tokenizer)
        self.token_searcher = FuzzyTokenSearcher(phrase_list=[phrase], tokenizer=self.tokenizer)
        doc = self.tokenizer.tokenize(doc_text='This is a test best case')
        token_matches = self.token_searcher.find_skipgram_token_matches_in_text(doc)
        for tm in token_matches:
            print(tm)
        candidates = get_partial_phrases(token_matches, self.token_searcher)
        for pi, phrase in enumerate(candidates):
            print('test_search_token_searcher - phrase:', phrase)
            with self.subTest(pi):
                for partial in candidates[phrase]:
                    print('\ntest_search_token_searcher - partial:', partial)
                    self.assertEqual(0, len(partial.missing_tokens))
                self.assertEqual(2, len(candidates[phrase]))
        # self.assertEqual(2, len(partial_match.text_tokens))


class TestFindMatches(TestCase):

    def setUp(self) -> None:
        self.tokenizer = Tokenizer()
        self.token_searcher = FuzzyTokenSearcher(phrase_list=['test case'], tokenizer=self.tokenizer)
        self.text = 'This is a test case'
        self.phrase = Phrase(self.text, tokenizer=self.tokenizer)

    def test_find_matches_accepts_text_string(self):
        error = None
        try:
            self.token_searcher.find_matches(self.text)
        except BaseException as err:
            error = err
        self.assertEqual(None, error)

    def test_find_matches_accepts_text_dictionary(self):
        error = None
        try:
            self.token_searcher.find_matches({'text': self.text, 'id': 'some_id'})
        except BaseException as err:
            error = err
        self.assertEqual(None, error)

    def test_find_matches_accepts_text_doc(self):
        error = None
        doc = self.tokenizer.tokenize(self.text, doc_id='some_id')
        try:
            self.token_searcher.find_matches(doc)
        except BaseException as err:
            error = err
        self.assertEqual(None, error)

    def test_making_partial_match_picks_best_phrase_option(self):
        self.token_searcher = FuzzyTokenSearcher(phrase_list=['best case', 'test case'], tokenizer=self.tokenizer)
        doc = self.tokenizer.tokenize(doc_text='This is a best test case')
        phrase_matches = self.token_searcher.find_matches(doc)
        for pi, phrase_match in enumerate(phrase_matches):
            print('test_search_token_searcher - phrase_match:', phrase_match)
            print('\t', phrase_match.levenshtein_similarity)
        self.assertEqual(1, len(phrase_matches))

    def test_find_phrase_matches_finds_best_option_per_text_range(self):
        self.token_searcher = FuzzyTokenSearcher(phrase_list=['best test case'], tokenizer=self.tokenizer)
        doc = self.tokenizer.tokenize(doc_text='this is a best test case to test best case matching')
        phrase_matches = self.token_searcher.find_matches(doc, debug=1)
        for pi, phrase_match in enumerate(phrase_matches):
            print('test_search_token_searcher - phrase_match:', phrase_match)
            print('\t', phrase_match.levenshtein_similarity)
            print('\t', phrase_match.offset, phrase_match.end)
        self.assertEqual(2, len(phrase_matches))

    def test_find_phrase_matches_filters_on_score_threshold(self):
        self.token_searcher = FuzzyTokenSearcher(phrase_list=['best test case'], tokenizer=self.tokenizer)
        doc = self.tokenizer.tokenize(doc_text='this is a best test case to test best case matching')
        self.token_searcher.config['levenshtein_threshold'] = 0.9
        phrase_matches = self.token_searcher.find_matches(doc, debug=1)
        for pi, phrase_match in enumerate(phrase_matches):
            print('test_search_token_searcher - phrase_match:', phrase_match)
            print('\t', phrase_match.levenshtein_similarity)
            print('\t', phrase_match.offset, phrase_match.end)
        for pi, pm in enumerate(phrase_matches):
            with self.subTest(pi):
                self.assertEqual(True, pm.levenshtein_similarity >= self.token_searcher.config['levenshtein_threshold'])


class TestTokenSearcherVocabulary(unittest.TestCase):

    def setUp(self) -> None:
        self.vocab = Vocabulary(['best', 'test'])
        self.tokenizer = Tokenizer()

    def test_searcher_can_have_vocab(self):
        token_searcher = FuzzyTokenSearcher(phrase_list=['best test case'], tokenizer=self.tokenizer,
                                            vocabulary=self.vocab)
        self.assertEqual(True, isinstance(token_searcher.vocabulary, Vocabulary))

    def test_searcher_can_match_tokens_via_vocab(self):
        token_searcher = FuzzyTokenSearcher(phrase_list=['best test case'], tokenizer=self.tokenizer,
                                            vocabulary=self.vocab)
        doc = self.tokenizer.tokenize('this is a test best case')
        matches = token_searcher.find_matches(doc)
        self.assertEqual(0, len(matches))


class TestTokenSearcherVariants(unittest.TestCase):

    def setUp(self) -> None:
        phrase_model = [
            {
                'phrase': 'best',
                'variants': ['rest']
            }
        ]
        self.searcher = FuzzyTokenSearcher(phrase_model=phrase_model)

    def test_searcher_can_match_phrase_variants(self):
        # TO DO: check that variants are matched and returned with their main phrase as match phrase
        self.assertEqual(True, False)

    """
"""
