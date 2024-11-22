import copy
import re
from collections import defaultdict
from typing import Callable, Dict, List, Set, Union


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
                 metadata: Dict[str, any] = None):
        """
        Initializes the Token instance.

        Args:
            string (str): The original string of the token.
            index (int): The index of the token in the document.
            char_index (int): The character index of the token in the document.
            doc_id (str, optional): The ID of the document the token belongs to.
            normalised_string (str, optional): The normalized version of the token.
            label (str, set, list, optional): Labels associated with the token.
            metadata (dict, optional): Additional metadata associated with the token.
        """
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

    def get_token(self, token_string: str) -> Union[Token, List[Token]]:
        """
        Retrieves a token (or tokens) from the document based on the token string.

        Args:
            token_string (str): The token string to look up.

        Returns:
            Union[Token, List[Token]]: A Token object or a list of Token objects that match the token string.
        """
        if token_string in self.token_orig_set:
            token = self.token_orig_set[token_string]
        elif token_string in self.token_norm_set:
            token = self.token_norm_set[token_string]
        else:
            token = None
        if isinstance(token, List) and len(token) == 1:
            token = token[0]
        return token

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
                 remove_punctuation: bool = False):
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

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text into a list of strings.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        non_whitespace_tokens = [token for token in text.strip().split() if token != '']
        tokens = []
        for nw_token in non_whitespace_tokens:
            if self.remove_punctuation:
                sub_tokens = re.split(r'\W+', nw_token)
            else:
                sub_tokens = re.split(r'(\W+)', nw_token)
            tokens.extend([token for token in sub_tokens if token != ''])
        return tokens

    def tokenize(self, doc_text: str, doc_id: str = None) -> List[Token]:
        """
        Tokenizes the input document text and returns a Doc object.

        Args:
            doc_text (str): The text of the document to tokenize.
            doc_id (str, optional): The ID of the document. Defaults to None.

        Returns:
            Doc: A Doc object containing the tokenized text.
        """
        dummy_text = f"<START> {doc_text} <END>" if self.include_boundary_tokens else doc_text
        token_strings = self._tokenize(doc_text)
        if self.include_boundary_tokens:
            token_strings = ['<START>'] + token_strings + ['<END>']
        tokens = []
        prefix_text = ''
        for ti, token_string in enumerate(token_strings):
            char_index = dummy_text.index(token_string) + len(prefix_text)
            prefix_text += dummy_text[:dummy_text.index(token_string) + len(token_string)]
            dummy_text = dummy_text[dummy_text.index(token_string)+len(token_string):]
            token = Token(token_string, index=ti, char_index=char_index, doc_id=doc_id)
            if self.ignorecase:
                if not self.include_boundary_tokens or (ti != 0 and ti != len(token_strings) - 1):
                    token.lower()
            tokens.append(token)
        return tokens

    def tokenize_doc(self, doc_text: str, doc_id: str = None):
        tokens = self.tokenize(doc_text, doc_id)
        return Doc(text=doc_text, doc_id=doc_id, tokens=tokens)


class RegExTokenizer(Tokenizer):
    """
    A tokenizer that splits text into tokens using a regular expression pattern.

    Attributes:
        split_pattern (str): The regular expression pattern used to split the text into tokens.
    """

    def __init__(self, ignorecase: bool = False, include_boundary_tokens: bool = False,
                 split_pattern: str = r'\b'):
        """
        Initializes the RegExTokenizer instance.

        Args:
            ignorecase (bool, optional): Whether to ignore case when tokenizing. Defaults to False.
            include_boundary_tokens (bool, optional): Whether to include boundary tokens. Defaults to False.
            split_pattern (str, optional): The regular expression pattern used to split the text. Defaults to r'\b'.
        """
        super().__init__(ignorecase=ignorecase, include_boundary_tokens=include_boundary_tokens)
        self.split_pattern = re.compile(split_pattern)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text into a list of strings using the specified regular expression pattern.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens obtained by splitting the input text.
        """
        # Split the text using the regular expression pattern
        tokens = [token.strip() for token in re.split(self.split_pattern, text) if token.strip() != '']
        non_whitespace_tokens = []
        for token in tokens:
            # Split multi-word tokens (e.g., if there is space between parts)
            split_tokens = token.split(' ')
            non_whitespace_tokens.extend([token for token in split_tokens if token != ''])
        return non_whitespace_tokens


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
                 metadata=copy.deepcopy(token.metadata))


def tokens2string(tokens: List[Token]) -> str:
    string = ''
    for token in tokens:
        if token.char_index > len(string):
            diff = token.char_index - len(string)
            string += ' ' * diff
        string += token.t
    return string
