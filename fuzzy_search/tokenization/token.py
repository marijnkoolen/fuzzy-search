import copy
import re
from collections import defaultdict
from typing import Callable, Dict, List, Set, Union


class Token:

    def __init__(self, string: str, index: int, char_index: int, doc_id: str = None,
                 normalised_string: str = None, label: Union[str, Set[str]] = None,
                 metadata: Dict[str, any] = None):
        self.string = string
        self.index = index
        self.char_index = char_index
        self.doc_id = doc_id
        self.metadata = metadata if metadata else {}
        self.normalised_string = normalised_string if normalised_string else string
        if label is None:
            label = set()
        elif isinstance(label, str):
            label = {label}
        elif isinstance(label, list):
            label = set(label)
        elif isinstance(label, set) is False:
            raise TypeError(f'token label must be of type "str", "set" or "list", not {type(label)}')
        self.label = label

    def __repr__(self):
        return f"'{self.normalised_string}'"

    def __len__(self):
        return len(self.normalised_string)

    def lower(self):
        self.normalised_string = self.normalised_string.lower()

    @property
    def i(self):
        return self.index

    @property
    def t(self):
        return self.string

    @property
    def n(self):
        return self.normalised_string

    @property
    def l(self):
        return self.label

    def has_label(self, label: str):
        return label in self.label

    def update(self, normalised_string: str):
        return Token(string=self.t, index=self.i, char_index=self.char_index,
                     normalised_string=normalised_string,
                     metadata=copy.deepcopy(self.metadata))


class Doc:

    def __init__(self, text: str, doc_id: str, tokens: List[Token], metadata: Dict[str, any] = None):
        self.text = text
        self.id = doc_id
        self.tokens = tokens
        self.label_token_index = defaultdict(set)
        for token in tokens:
            for label in token.label:
                self.label_token_index[label].add(token)
        self.metadata = metadata if metadata else {}

    def __repr__(self):
        return f"Doc(id='{self.id}', metadata={self.metadata}, text=\"{self.text}\", tokens={self.tokens}"

    def __len__(self):
        return len(self.normalized)

    def __getitem__(self, item):
        return self.tokens[item]

    def __iter__(self):
        for token in self.tokens:
            yield token

    @property
    def original(self):
        return [token.t for token in self.tokens]

    @property
    def normalized(self):
        return [token.n for token in self.tokens]


class Tokenizer:

    def __init__(self, ignorecase: bool = False, include_boundary_tokens: bool = False):
        self.ignorecase = ignorecase
        self.include_boundary_tokens = include_boundary_tokens

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in text.strip().split() if token != '']

    def tokenize(self, doc_text: str, doc_id: str = None) -> Doc:
        if self.include_boundary_tokens:
            doc_text = f"<START> {doc_text} <END>"
        token_strings = self._tokenize(doc_text)
        # if self.include_boundary_tokens:
        #     tokens = ['<START>'] + token_strings + ['<END>']
        # tokens = [Token(token, index=ti, doc_id=doc_id) for ti, token in enumerate(tokens)]
        tokens = []
        dummy_text = doc_text
        prefix_text = ''
        for ti, token_string in enumerate(token_strings):
            # print('tokenize - ti, token_string:', ti, token_string)
            # print('tokenize - prefix_text:', prefix_text)
            char_index = dummy_text.index(token_string) + len(prefix_text)
            # print('tokenize - char_index:', char_index)
            prefix_text += dummy_text[:dummy_text.index(token_string) + len(token_string)]
            dummy_text = dummy_text[dummy_text.index(token_string)+len(token_string):]
            # print('tokenize - dummy_text:', dummy_text)
            token = Token(token_string, index=ti, char_index=char_index, doc_id=doc_id)
            # print('tokenize - token.char_index:', token.char_index)
            tokens.append(token)
        if self.ignorecase:
            for t in tokens:
                t.lower()
        return Doc(text=doc_text, doc_id=doc_id, tokens=tokens)


class RegExTokenizer(Tokenizer):

    def __init__(self, ignorecase: bool = False, include_boundary_tokens: bool = False,
                 split_pattern: str = r'\b'):
        super().__init__(ignorecase=ignorecase, include_boundary_tokens=include_boundary_tokens)
        self.split_pattern = re.compile(split_pattern)

    def _tokenize(self, text: str) -> List[str]:
        return [token.replace(' ', '') for token in re.split(self.split_pattern, text) if token.replace(' ', '') != '']
        # return [token for token in re.split(r'\W+', text) if token != '']


class CustomTokenizer(Tokenizer):

    def __init__(self, tokenizer_func: Callable, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer_func = tokenizer_func

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenizer_func(text)


def update_token(token: Token, new_normalised: str) -> Token:
    return Token(string=token.t, index=token.i, char_index=token.char_index,
                 normalised_string=new_normalised,
                 metadata=copy.deepcopy(token.metadata))
