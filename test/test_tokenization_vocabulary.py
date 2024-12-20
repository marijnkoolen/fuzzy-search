from unittest import TestCase

from fuzzy_search.tokenization.token import Tokenizer
from fuzzy_search.tokenization.vocabulary import Vocabulary


class TestVocabulary(TestCase):

    def setUp(self) -> None:
        self.words = ['This', 'is', 'a', 'list', 'of', 'words']
        text = ' '.join(self.words)
        tokenizer = Tokenizer()
        self.doc = tokenizer.tokenize_doc(text)

    def test_vocabulary_can_be_initialised_empty(self):
        vocab = Vocabulary()
        self.assertEqual(0, len(vocab.term_id))

    def test_vocabulary_can_be_initialised_with_word_list(self):
        vocab = Vocabulary(self.words)
        self.assertEqual(len(self.words), len(vocab.term_id))

    def test_vocabulary_can_be_initialised_with_token_list(self):
        vocab = Vocabulary(self.doc.tokens)
        self.assertEqual(len(self.words), len(vocab.term_id))

    def test_vocabulary_represents_words_by_integer(self):
        vocab = Vocabulary(self.words)
        self.assertEqual(True, isinstance(vocab.term_id[self.words[0]], int))

    def test_vocabulary_can_check_that_word_is_included(self):
        vocab = Vocabulary(self.words)
        self.assertEqual(True, vocab.has_term('This'))

    def test_vocabulary_can_use_contains_dunder(self):
        vocab = Vocabulary(self.words)
        self.assertEqual(True, 'This' in vocab)

    def test_vocabulary_can_check_that_token_is_included(self):
        vocab = Vocabulary(self.doc.tokens)
        self.assertEqual(True, vocab.has_term(self.doc.tokens[0]))

    def test_vocabulary_can_be_case_insensitive(self):
        vocab = Vocabulary(self.words, ignorecase=True)
        self.assertEqual(True, vocab.has_term('this'))

    def test_vocabulary_can_be_case_sensitive(self):
        vocab = Vocabulary(self.words, ignorecase=False)
        self.assertEqual(True, vocab.has_term('This'))
        self.assertEqual(False, vocab.has_term('this'))
