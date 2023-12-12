import copy
import re
from collections import defaultdict
from typing import Callable, Dict, List, Set, Union


class Annotation:

    def __init__(self, tag_type: str, text: str, offset: int, doc_id: str = None):
        self.tag_type = tag_type
        self.text = text
        self.offset = offset
        self.end = offset + len(text)
        self.doc_id = doc_id


class Tag:

    def __init__(self, tag_type: str, text: str, offset: int, doc_id: str = None):
        self.type = tag_type
        self.text = text
        self.offset = offset
        self.tag_string = f"<{tag_type}>{text}</{tag_type}>"
        self.end = offset + len(self.tag_string)
        self.doc_id = doc_id


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
        self.token_orig_set = {}
        self.token_norm_set = {}
        self.label_token_index = defaultdict(set)
        for token in tokens:
            for label in token.label:
                self.label_token_index[label].add(token)
        self.metadata = metadata if metadata else {}
        self.annotations: List[Annotation] = []
        for token in tokens:
            if token.t not in self.token_orig_set:
                self.token_orig_set[token.t] = [token]
            else:
                self.token_orig_set[token.t].append(token)
            if token.n not in self.token_norm_set:
                self.token_norm_set[token.n] = [token]
            else:
                self.token_norm_set[token.n].append(token)

    def __repr__(self):
        return f"Doc(id='{self.id}', metadata={self.metadata}, text=\"{self.text}\", tokens={self.tokens}"

    def __len__(self):
        return len(self.normalized)

    def __getitem__(self, item):
        return self.tokens[item]

    def __iter__(self):
        for token in self.tokens:
            yield token

    def _has_original_token(self, token: str) -> bool:
        return token in self.token_orig_set

    def _has_normalised_token(self, token: str) -> bool:
        return token in self.token_norm_set

    def has_token(self, token: Union[str, Token]) -> bool:
        if isinstance(token, str):
            token = self.get_token(token)
            return token is not None
        else:
            return self._has_original_token(token.t) or self._has_normalised_token(token.n)

    def get_token(self, token_string: str) -> Union[Token, List[Token]]:
        if token_string in self.token_orig_set:
            token = self.token_orig_set[token_string]
        elif token_string in self.token_norm_set:
            token = self.token_norm_set[token_string]
        else:
            token = None
        if isinstance(token, List) and len(token) == 1:
            token = token[0]
        return token

    def add_annotations(self, annotations: List[Annotation]):
        self.annotations = annotations

    @property
    def original(self):
        return [token.t for token in self.tokens]

    @property
    def normalized(self):
        return [token.n for token in self.tokens]


class Tokenizer:

    def __init__(self, ignorecase: bool = False, include_boundary_tokens: bool = False,
                 remove_punctuation: bool = False):
        self.ignorecase = ignorecase
        self.include_boundary_tokens = include_boundary_tokens
        self.remove_punctuation = remove_punctuation

    def _tokenize(self, text: str) -> List[str]:
        non_whitespace_tokens = [token for token in text.strip().split() if token != '']
        tokens = []
        for nw_token in non_whitespace_tokens:
            if self.remove_punctuation:
                sub_tokens = re.split(r'\W+', nw_token)
            else:
                sub_tokens = re.split(r'(\W+)', nw_token)
            tokens.extend([token for token in sub_tokens if token != ''])
        return tokens

    def tokenize(self, doc_text: str, doc_id: str = None) -> Doc:
        dummy_text = f"<START> {doc_text} <END>" if self.include_boundary_tokens else doc_text
        token_strings = self._tokenize(doc_text)
        if self.include_boundary_tokens:
            token_strings = ['<START>'] + token_strings + ['<END>']
        # tokens = [Token(token, index=ti, doc_id=doc_id) for ti, token in enumerate(tokens)]
        tokens = []
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
            if self.ignorecase:
                if not self.include_boundary_tokens or (ti != 0 and ti != len(token_strings) - 1):
                    token.lower()
            tokens.append(token)
        return Doc(text=doc_text, doc_id=doc_id, tokens=tokens)


class RegExTokenizer(Tokenizer):

    def __init__(self, ignorecase: bool = False, include_boundary_tokens: bool = False,
                 split_pattern: str = r'\b'):
        super().__init__(ignorecase=ignorecase, include_boundary_tokens=include_boundary_tokens)
        self.split_pattern = re.compile(split_pattern)

    def _tokenize(self, text: str) -> List[str]:
        # print(re.split(self.split_pattern, text))
        tokens = [token.strip() for token in re.split(self.split_pattern, text) if token.strip() != '']
        non_whitespace_tokens = []
        for token in tokens:
            split_tokens = token.split(' ')
            non_whitespace_tokens.extend([token for token in split_tokens if token != ''])
        # print('non_whitespace_tokens:', non_whitespace_tokens)
        return non_whitespace_tokens
        # return [token.replace(' ', '') for token in re.split(self.split_pattern, text) if token.replace(' ', '') != '']
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
