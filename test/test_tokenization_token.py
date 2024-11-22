from unittest import TestCase

from fuzzy_search.tokenization.token import Token
from fuzzy_search.tokenization.token import Doc
from fuzzy_search.tokenization.token import Tokenizer
from fuzzy_search.tokenization.token import CustomTokenizer
from fuzzy_search.tokenization.token import tokens2string


class TestToken(TestCase):

    def test_tokenizer_token_has_normalised_string(self):
        token = Token(string='test', index=0, char_index=0)
        self.assertEqual(token.string, token.normalised_string)

    def test_tokenizer_token_can_have_label_string(self):
        token = Token(string='test', index=0, char_index=0, label='test_label')
        self.assertEqual(True, 'test_label' in token.label)

    def test_tokenizer_token_can_have_label_set(self):
        token = Token(string='test', index=0, char_index=0, label={'test', 'label'})
        self.assertEqual(True, 'test' in token.label)

    def test_tokenizer_token_can_check_has_label(self):
        token = Token(string='test', index=0, char_index=0, label={'test', 'label'})
        self.assertEqual(True, token.has_label('label'))

    def test_tokenizer_lower_affects_normalised_string(self):
        token = Token(string='Test', index=0, char_index=0)
        token.lower()
        self.assertEqual('Test', token.string)
        self.assertEqual('test', token.normalised_string)

    def test_tokenizer_update_affects_normalised_string(self):
        token = Token(string='test', index=0, char_index=0)
        updated_token = token.update('text')
        self.assertEqual('text', updated_token.normalised_string)
        self.assertEqual('test', updated_token.string)

    def test_tokenizer_update_returns_new_token(self):
        token = Token(string='test', index=0, char_index=0)
        updated_token = token.update('text')
        self.assertEqual('test', token.normalised_string)
        self.assertEqual('text', updated_token.normalised_string)


class TestTokenizer(TestToken):

    def setUp(self) -> None:
        self.text = 'This is an example sentence.'

    def test_tokenize_returns_a_list_of_tokens(self):
        tokenizer = Tokenizer(include_boundary_tokens=False)
        tokens = tokenizer.tokenize(self.text)
        for ti, token in enumerate(tokens):
            with self.subTest(ti):
                self.assertEqual(True, isinstance(token, Token))

    def test_tokenize_doc_returns_a_document_object(self):
        tokenizer = Tokenizer(include_boundary_tokens=False)
        doc = tokenizer.tokenize_doc(self.text)
        self.assertEqual(True, isinstance(doc, Doc))

    def test_tokenizer_tracks_char_index(self):
        tokenizer = Tokenizer(include_boundary_tokens=False)
        tokens = tokenizer.tokenize(self.text)
        token = tokens[3]
        self.assertEqual(self.text.index(token.t), token.char_index)

    def test_tokenizer_defaults_to_no_boundary_tokens(self):
        tokenizer = Tokenizer(include_boundary_tokens=False)
        tokens = tokenizer.tokenize(self.text)
        self.assertEqual(False, tokens[0].string == '<START>')

    def test_tokenizer_can_add_boundaries(self):
        tokenizer = Tokenizer(include_boundary_tokens=True)
        tokens = tokenizer.tokenize(self.text)
        self.assertEqual(True, tokens[0].string == '<START>')

    def test_tokenizer_ignorecase_keeps_original_case_in_string(self):
        tokenizer = Tokenizer(ignorecase=True)
        tokens = tokenizer.tokenize(self.text)
        self.assertEqual(True, tokens[0].string == 'This')

    def test_tokenizer_ignorecase_lowercases_normalised_string(self):
        tokenizer = Tokenizer(ignorecase=True)
        tokens = tokenizer.tokenize(self.text)
        self.assertEqual(True, tokens[0].normalised_string == 'this')


class TestCustomTokenizer(TestCase):

    def setUp(self) -> None:
        self.text = 'This is an example sentence.'

    def test_any_string_tokenizing_func_returns_doc(self):
        def tokenize_func(text):
            return text.split(' ')
        tokenizer = CustomTokenizer(tokenizer_func=tokenize_func)
        tokens = tokenize_func(self.text)
        custom_tokens = tokenizer.tokenize(self.text)
        for ti, token in enumerate(tokens):
            with self.subTest(ti):
                self.assertEqual(token, custom_tokens[ti].t)


class TestToken2String(TestCase):

    def setUp(self) -> None:
        self.string = "Yeah, well, you know, that's just like, uh, your opinion, man."
        self.tokenizer = Tokenizer()
        self.tokens = self.tokenizer.tokenize(self.string)

    def test_text2string_can_reconstruct_original_string(self):
        new_string = tokens2string(self.tokens)
        self.assertEqual(new_string, self.string)

