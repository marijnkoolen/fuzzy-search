from unittest import TestCase

from fuzzy_search.tokenization.token import Token
from fuzzy_search.tokenization.token import Doc
from fuzzy_search.tokenization.token import Tokenizer
from fuzzy_search.tokenization.token import RegExTokenizer
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

    def test_tokenizer_tracks_char_end_index(self):
        tokenizer = Tokenizer(include_boundary_tokens=False)
        tokens = tokenizer.tokenize(self.text)
        token = tokens[3]
        token_start_index = self.text.index(token.t)
        token_end_index = token_start_index + len(token) + 1
        doc_end_index = len(self.text) - token_end_index
        self.assertEqual(doc_end_index, token.char_end_index)

    def test_tokenizer_defaults_to_no_boundary_tokens(self):
        tokenizer = Tokenizer(include_boundary_tokens=False)
        tokens = tokenizer.tokenize(self.text)
        self.assertEqual(False, tokens[0].string == '<DOC>')

    def test_tokenizer_can_add_boundaries(self):
        tokenizer = Tokenizer(include_boundary_tokens=True)
        tokens = tokenizer.tokenize(self.text)
        self.assertEqual(True, tokens[0].string == '<DOC>')

    def test_tokenizer_ignorecase_keeps_original_case_in_string(self):
        tokenizer = Tokenizer(ignorecase=True)
        tokens = tokenizer.tokenize(self.text)
        self.assertEqual(True, tokens[0].string == 'This')

    def test_tokenizer_ignorecase_lowercases_normalised_string(self):
        tokenizer = Tokenizer(ignorecase=True)
        tokens = tokenizer.tokenize(self.text)
        self.assertEqual(True, tokens[0].normalised_string == 'this')


class TestRegexTokenizer(TestCase):

    def setUp(self) -> None:
        self.text = 'This is an example sentence.'
        self.split_tokenizer = RegExTokenizer(split_pattern=r"\s+")
        self.token_tokenizer = RegExTokenizer(token_pattern=r"\w+")

    def test_regex_tokenizing_split_func_returns_correct_tokens(self):
        import re
        def tokenize_func(text):
            return re.split(r'\s+', text)
        regex_tokens = self.split_tokenizer.tokenize(self.text)
        tokens = [token for token in tokenize_func(self.text) if token != '']
        print(tokens)
        print(regex_tokens)
        for ti, token in enumerate(tokens):
            with self.subTest(ti):
                self.assertEqual(token, regex_tokens[ti].t)

    def test_regex_tokenizing_split_func_returns_correct_char_index(self):
        def tokenize_func(text):
            return re.split(r'\s+', text)
        tokens = self.split_tokenizer.tokenize(self.text)
        for ti, token in enumerate(tokens):
            with self.subTest(ti):
                self.assertEqual(self.text[token.char_index:token.char_index+len(token)], token.t)

    def test_regex_tokenizing_token_func_returns_correct_tokens(self):
        import re
        def tokenize_func(text):
            return [m.group(0) for m in re.finditer(r'\w+', text)]
        regex_tokens = self.token_tokenizer.tokenize(self.text)
        tokens = [token for token in tokenize_func(self.text) if token != '']
        print(tokens)
        print(regex_tokens)
        for ti, token in enumerate(tokens):
            with self.subTest(ti):
                self.assertEqual(token, regex_tokens[ti].t)

    def test_regex_tokenizing_token_func_returns_correct_char_index(self):
        def tokenize_func(text):
            return [m.group(0) for m in re.finditer(r'\w+', text)]
        tokens = self.token_tokenizer.tokenize(self.text)
        for ti, token in enumerate(tokens):
            with self.subTest(ti):
                self.assertEqual(self.text[token.char_index:token.char_index+len(token)], token.t)


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

    def test_any_string_tokenizing_func_returns_correct_char_index(self):
        def tokenize_func(text):
            return text.split(' ')
        tokenizer = CustomTokenizer(tokenizer_func=tokenize_func)
        custom_tokens = tokenizer.tokenize(self.text)
        for ti, token in enumerate(custom_tokens):
            with self.subTest(ti):
                self.assertEqual(self.text[token.char_index:token.char_index+len(token)], token.t)


class TestToken2String(TestCase):

    def setUp(self) -> None:
        self.string = "Yeah, well, you know, that's just like, uh, your opinion, man."
        self.tokenizer = Tokenizer()
        self.tokens = self.tokenizer.tokenize(self.string)

    def test_text2string_can_reconstruct_original_string(self):
        new_string = tokens2string(self.tokens)
        self.assertEqual(new_string, self.string)

