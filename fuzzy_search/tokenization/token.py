import copy
import re
from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple, Union

from nltk.tokenize import WordPunctTokenizer


class Annotation:
    """
    Represents an annotation in a document.

    Attributes:
        tag_type (str): The type of the tag.
        text (str): The text of the annotation.
        offset (int): The starting index of the annotation.
        end (int): The ending index of the annotation (offset + length of the text).
        doc_id (str, optional): The ID of the document the annotation belongs to.
    """

    def __init__(self, tag_type: str, text: str, offset: int, doc_id: str = None):
        """
        Initializes the Annotation instance.

        Args:
            tag_type (str): The type of the tag.
            text (str): The text of the annotation.
            offset (int): The starting index of the annotation.
            doc_id (str, optional): The ID of the document the annotation belongs to.
        """
        self.tag_type = tag_type
        self.text = text
        self.offset = offset
        self.end = offset + len(text)
        self.doc_id = doc_id


class Tag:
    """
    Represents a tag in a document, containing both the tag's type and text.

    Attributes:
        type (str): The type of the tag.
        text (str): The text content of the tag.
        offset (int): The starting index of the tag in the document.
        tag_string (str): The full tag string, including opening and closing tags.
        end (int): The ending index of the tag (offset + length of the tag string).
        doc_id (str, optional): The ID of the document the tag belongs to.
    """

    def __init__(self, tag_type: str, text: str, offset: int, doc_id: str = None):
        """
        Initializes the Tag instance.

        Args:
            tag_type (str): The type of the tag.
            text (str): The text content of the tag.
            offset (int): The starting index of the tag.
            doc_id (str, optional): The ID of the document the tag belongs to.
        """
        self.type = tag_type
        self.text = text
        self.offset = offset
        self.tag_string = f"<{tag_type}>{text}</{tag_type}>"
        self.end = offset + len(self.tag_string)
        self.doc_id = doc_id


class Token:
    """
    Represents a token in a document.

    Attributes:
        string (str): The original string of the token.
        index (int): The index of the token in the document.
        char_index (int): The character index of the token in the document.
        doc_id (str, optional): The ID of the document the token belongs to.
        metadata (dict, optional): Additional metadata associated with the token.
        normalised_string (str): The normalized version of the token.
        label (set): A set of labels assigned to the token.
    """

    def __init__(self, string: str, index: int, char_index: int, doc_id: str = None,
                 normalised_string: str = None, label: Union[str, Set[str]] = None,
                 char_end_index: int = None, metadata: Dict[str, any] = None):
        """
        Initializes the Token instance.

        Args:
            string (str): The original string of the token.
            index (int): The index of the token in the document.
            char_index (int): The character index of the token in the document.
            doc_id (str, optional): The ID of the document the token belongs to.
            normalised_string (str, optional): The normalized version of the token.
            label (str, set, list, optional): Labels associated with the token.
            char_index (int): The character index of the token from the end of the document.
            metadata (dict, optional): Additional metadata associated with the token.
        """
        self.string = string
        self.index = index
        self.char_index = char_index
        self.char_end_index = char_end_index
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
        """
        Returns a string representation of the Token.

        Returns:
            str: A string representation of the token's normalized string.
        """
        return f"'{self.normalised_string}'"

    def __len__(self):
        """
        Returns the length of the token.

        Returns:
            int: The length of the normalized string of the token.
        """
        return len(self.normalised_string)

    def lower(self):
        """
        Converts the normalized string of the token to lowercase.
        """
        self.normalised_string = self.normalised_string.lower()

    @property
    def i(self):
        """
        Gets the index of the token.

        Returns:
            int: The index of the token.
        """
        return self.index

    @property
    def t(self):
        """
        Gets the original string of the token.

        Returns:
            str: The original string of the token.
        """
        return self.string

    @property
    def n(self):
        """
        Gets the normalized string of the token.

        Returns:
            str: The normalized string of the token.
        """
        return self.normalised_string

    @property
    def l(self):
        """
        Gets the labels associated with the token.

        Returns:
            set: A set of labels associated with the token.
        """
        return self.label

    def has_label(self, label: str):
        """
        Checks if the token has a specific label.

        Args:
            label (str): The label to check for.

        Returns:
            bool: True if the token has the label, False otherwise.
        """
        return label in self.label

    def update(self, normalised_string: str) -> 'Token':
        """
        Creates a new token with an updated normalized string.

        Args:
            normalised_string (str): The updated normalized string.

        Returns:
            Token: A new token with the updated normalized string.
        """
        return Token(string=self.t, index=self.i, char_index=self.char_index,
                     normalised_string=normalised_string,
                     metadata=copy.deepcopy(self.metadata))


class Doc:
    """
    Represents a document containing a list of tokens.

    Attributes:
        text (str): The text of the document.
        id (str): The ID of the document.
        tokens (list): A list of Token objects representing the document's tokens.
        token_orig_set (dict): A dictionary mapping original token strings to lists of tokens.
        token_norm_set (dict): A dictionary mapping normalized token strings to lists of tokens.
        label_token_index (defaultdict): A dictionary mapping labels to sets of tokens.
        metadata (dict): Metadata associated with the document.
        annotations (list): A list of annotations associated with the document.
    """

    def __init__(self, text: str, doc_id: str, tokens: List[Token], metadata: Dict[str, any] = None):
        """
        Initializes a Document instance.

        Args:
            text (str): The text content of the document.
            doc_id (str): The ID of the document.
            tokens (List[Token]): A list of Token objects representing the document's tokens.
            metadata (dict, optional): Metadata associated with the document.
        """
        self.text = text
        self.id = doc_id
        self.tokens = tokens
        self.token_orig_set = {}
        self.token_norm_set = {}
        self.label_token_index = defaultdict(set)
        self.metadata = metadata if metadata else {}
        self.annotations: List[Annotation] = []
        for token in tokens:
            for label in token.label:
                self.label_token_index[label].add(token)
            if token.t not in self.token_orig_set:
                self.token_orig_set[token.t] = [token]
            else:
                self.token_orig_set[token.t].append(token)
            if token.n not in self.token_norm_set:
                self.token_norm_set[token.n] = [token]
            else:
                self.token_norm_set[token.n].append(token)

    def __repr__(self):
        """
        Returns a string representation of the document.

        Returns:
            str: A string representation of the document, including its ID, metadata, and tokens.
        """
        return f"Doc(id='{self.id}', metadata={self.metadata}, text=\"{self.text}\", tokens={self.tokens}"

    def __len__(self):
        """
        Returns the number of normalized tokens in the document.

        Returns:
            int: The number of normalized tokens in the document.
        """
        return len(self.normalized)

    def __getitem__(self, item):
        """
        Retrieves a token by index.

        Args:
            item (int): The index of the token to retrieve.

        Returns:
            Token: The token at the specified index.
        """
        return self.tokens[item]

    def __iter__(self):
        """
        Iterates over the tokens in the document.

        Yields:
            Token: Each token in the document.
        """
        for token in self.tokens:
            yield token

    def _has_original_token(self, token: str) -> bool:
        """
        Checks if the token exists in the original set of tokens.

        Args:
            token (str): The original token string.

        Returns:
            bool: True if the token exists in the original set, False otherwise.
        """
        return token in self.token_orig_set

    def _has_normalised_token(self, token: str) -> bool:
        """
        Checks if the token exists in the normalized set of tokens.

        Args:
            token (str): The normalized token string.

        Returns:
            bool: True if the token exists in the normalized set, False otherwise.
        """
        return token in self.token_norm_set

    def has_token(self, token: Union[str, Token]) -> bool:
        """
        Checks if the document contains a specific token.

        Args:
            token (Union[str, Token]): A token string or a Token object.

        Returns:
            bool: True if the token exists in the document, False otherwise.
        """
        if isinstance(token, str):
            token = self.get_token(token)
            return token is not None
        else:
            return self._has_original_token(token.t) or self._has_normalised_token(token.n)

    def get_token(self, token_string: str) -> Union[Token, List[Token], None]:
        """
        Retrieves a token (or tokens) from the document based on the token string.

        Args:
            token_string (str): The token string to look up.

        Returns:
            Union[Token, List[Token]]: A Token object or a list of Token objects that match the token string.
        """
        if token_string in self.token_orig_set:
            token_list = self.token_orig_set[token_string]
        elif token_string in self.token_norm_set:
            token_list = self.token_norm_set[token_string]
        else:
            return None
        if len(token_list) == 1:
            return token_list[0]
        return token_list

    def add_annotations(self, annotations: List[Annotation], replace: bool = False):
        """
        Adds annotations to the document.

        Args:
            annotations (List[Annotation]): A list of Annotation objects to add.
            replace (bool): whether to replace existing annotation or add (default is False).
        """
        self.annotations = annotations

    @property
    def original(self):
        """
        Retrieves a list of original tokens in the document.

        Returns:
            List[str]: A list of the original tokens.
        """
        return [token.t for token in self.tokens]

    @property
    def normalized(self):
        """
        Retrieves a list of normalized tokens in the document.

        Returns:
            List[str]: A list of the normalized tokens.
        """
        return [token.n for token in self.tokens]


class Tokenizer:
    """
    A base class for tokenizing a document into tokens.

    Attributes:
        ignorecase (bool): Flag indicating whether to ignore case when tokenizing.
        include_boundary_tokens (bool): Flag indicating whether to include boundary tokens.
        remove_punctuation (bool): Flag indicating whether to remove punctuation from tokens.
    """

    def __init__(self, ignorecase: bool = False, include_boundary_tokens: bool = False,
                 remove_punctuation: bool = False, split_pattern=r"(\S+)"):
        """
        Initializes the Tokenizer instance.

        Args:
            ignorecase (bool, optional): Whether to ignore case when tokenizing. Defaults to False.
            include_boundary_tokens (bool, optional): Whether to include boundary tokens. Defaults to False.
            remove_punctuation (bool, optional): Whether to remove punctuation. Defaults to False.
        """
        self.ignorecase = ignorecase
        self.include_boundary_tokens = include_boundary_tokens
        self.remove_punctuation = remove_punctuation
        self.split_pattern = re.compile(split_pattern)
        self.nltk_wp_tokenizer = WordPunctTokenizer()

    def _string_tokenizer(self, text) -> Tuple[str, int, int]:
        for si, token_span in enumerate(self.nltk_wp_tokenizer.span_tokenize(text)):
            token_string = text[token_span[0]:token_span[1]]
            if self.remove_punctuation is True and token_string.isalnum() is False:
                continue
            char_index = token_span[0]
            yield token_string, char_index

    def _tokenize(self, text: str) -> List[Union[str, Token]]:
        """
        Tokenizes the input text into a list of strings.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        tokens = []
        doc_length = len(text)
        if self.include_boundary_tokens is True:
            start_token = Token('<DOC>', index=0, char_index=0, char_end_index=doc_length, normalised_string='')
            tokens.append(start_token)
        for token_string, char_index in self._string_tokenizer(text):
            char_end_index = doc_length - (char_index + len(token_string) + 1)
            token = Token(token_string, index=len(tokens), char_index=char_index,
                          char_end_index=char_end_index)
            if self.ignorecase is True:
                token.lower()
            tokens.append(token)
        if self.include_boundary_tokens is True:
            end_token = Token('</DOC>', index=len(tokens), char_index=doc_length, char_end_index=0, normalised_string='')
            tokens.append(end_token)
        return tokens

    def tokenize(self, doc_text: str, doc_id: str = None) -> List[Token]:
        """
        Tokenizes the input document text and returns a list of documents.

        Args:
            doc_text (str): The text of the document to tokenize.
            doc_id (str, optional): The ID of the document. Defaults to None.

        Returns:
            Doc: A Doc object containing the tokenized text.
        """
        token_strings = self._tokenize(doc_text)
        if doc_id is not None:
            for token in token_strings:
                token.doc_id = doc_id
        return token_strings

    def tokenize_doc(self, doc_text: str, doc_id: str = None):
        """
        Tokenizes the input document text and returns a Doc object.

        Args:
            doc_text (str): The text of the document to tokenize.
            doc_id (str, optional): The ID of the document. Defaults to None.

        Returns:
            Doc: A Doc object containing the tokenized text.
        """
        tokens = self.tokenize(doc_text, doc_id)
        return Doc(text=doc_text, doc_id=doc_id, tokens=tokens)


class RegExTokenizer(Tokenizer):
    """
    A tokenizer that splits text into tokens using a regular expression pattern.

    Attributes:
        split_pattern (str): The regular expression pattern used to split the text into tokens.
    """

    def __init__(self, ignorecase: bool = False, include_boundary_tokens: bool = False,
                 split_pattern: str = r"\s+", token_pattern: str = None):
        """
        Initializes the RegExTokenizer instance.

        Args:
            ignorecase (bool, optional): Whether to ignore case when tokenizing. Defaults to False.
            include_boundary_tokens (bool, optional): Whether to include boundary tokens. Defaults to False.
            split_pattern (str, optional): The regular expression pattern used to split the text. Defaults to r'\b'.
        """
        super().__init__(ignorecase=ignorecase, include_boundary_tokens=include_boundary_tokens)
        self.remove_punctuation = False
        if token_pattern is not None:
            self.split_pattern = None
            self.token_pattern = re.compile(token_pattern)
            self._string_tokenizer = self._token_pattern_tokenizer
        else:
            self.split_pattern = re.compile(split_pattern)
            self.token_pattern = None
            self._string_tokenizer = self._split_pattern_tokenizer

    def _split_pattern_tokenizer(self, text: str):
        split_matches = [match for match in re.finditer(self.split_pattern, text)]
        char_index = 0
        for split_match in split_matches:
            token_end = split_match.start()
            token_string = text[char_index:token_end]
            if len(token_string) > 0:
                # only yield tokens that have at least one character
                yield token_string, char_index
            char_index = split_match.end()
        token_string = text[char_index:]
        yield token_string, char_index

    def _token_pattern_tokenizer(self, text: str):
        token_matches = [match for match in re.finditer(self.token_pattern, text)]
        for token_match in token_matches:
            yield token_match.group(0), token_match.start()


class CustomTokenizer(Tokenizer):
    """
    A tokenizer that uses a custom tokenizer function provided by the user.

    Attributes:
        tokenizer_func (Callable): A user-defined function for tokenizing text.
    """

    def __init__(self, tokenizer_func: Callable, **kwargs):
        """
        Initializes the CustomTokenizer instance.

        Args:
            tokenizer_func (Callable): The custom function to use for tokenizing the text.
            **kwargs: Additional arguments passed to the parent Tokenizer class.
        """
        super().__init__(**kwargs)
        self.tokenizer_func = tokenizer_func

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text using the custom tokenizer function.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens generated by the custom tokenizer function.
        """
        return self.tokenizer_func(text)

    def _strings_to_tokens(self, doc: Dict[str, any], token_strings: List[str]) -> List[Token]:
        # dummy_text = f"<doc> {doc['text']} </doc>" if self.include_boundary_tokens else doc['text']
        # if self.include_boundary_tokens:
        #     token_strings = ['<DOC>'] + token_strings + ['</DOC>']
        dummy_text = doc['text']
        tokens = []
        doc_length = len(doc['text'])
        if self.include_boundary_tokens is True:
            start_token = Token('<DOC>', index=0, char_index=0, char_end_index=doc_length, normalised_string='')
            tokens.append(start_token)
        prefix_text = ''
        for ti, token_string in enumerate(token_strings):
            dummy_char_index = dummy_text.index(token_string)
            char_index = dummy_char_index + len(prefix_text)
            char_end_index = doc_length - (char_index + len(token_string) + 1)
            prefix_text += dummy_text[:dummy_char_index + len(token_string)]
            dummy_text = dummy_text[dummy_char_index+len(token_string):]
            token = Token(token_string, index=len(tokens), char_index=char_index, doc_id=doc['id'],
                          char_end_index=char_end_index)
            if self.ignorecase:
                if not self.include_boundary_tokens or (ti != 0 and ti != len(token_strings) - 1):
                    token.lower()
            tokens.append(token)
        if self.include_boundary_tokens is True:
            end_token = Token('</DOC>', index=len(tokens), char_index=doc_length, char_end_index=0, normalised_string='')
            tokens.append(end_token)
        return tokens

    def tokenize(self, doc_text: str, doc_id: str = None) -> List[Token]:
        token_strings = self._tokenize(doc_text)
        doc = {'id': doc_id, 'text': doc_text}
        tokens = self._strings_to_tokens(doc, token_strings)
        return tokens


def update_token(token: Token, new_normalised: str) -> Token:
    """
    Creates a new Token instance by updating the normalization string of an existing token.

    Args:
        token (Token): The original Token object to be updated.
        new_normalised (str): The new normalized string for the token.

    Returns:
        Token: A new Token object with the updated normalization string.
    """
    return Token(string=token.t, index=token.i, char_index=token.char_index,
                 normalised_string=new_normalised,
                 metadata=copy.deepcopy(token.metadata),
                 char_end_index=token.char_end_index)


def tokens2string(tokens: List[Token]) -> str:
    string = ''
    for token in tokens:
        if token.char_index > len(string):
            diff = token.char_index - len(string)
            string += ' ' * diff
        string += token.t
    return string
