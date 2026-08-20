"""A simple bidirectional term-to-identifier mapping used for indexing terms and skipgrams."""

from typing import List, Union

from .token import Token


class Vocabulary:
    """Maps terms (strings or Tokens) to integer identifiers and back.

    Terms are assigned identifiers in the order they are first added, starting from 0.

    Attributes:
        term_id (Dict[str, int]): Maps each known term to its identifier.
        id_term (Dict[int, str]): Maps each identifier back to its term.
        term_freq (Dict[str, int]): Reserved for term frequency tracking (currently unused).
        ignorecase (bool): Whether terms are lowercased before being indexed/looked up.
    """

    def __init__(self, terms: List[Union[str, Token]] = None, ignorecase: bool = False):
        """Initializes the Vocabulary, optionally indexing an initial list of terms.

        Args:
            terms (List[Union[str, Token]], optional): Terms to add to the vocabulary
                on creation.
            ignorecase (bool, optional): Whether to lowercase terms before indexing
                and lookup. Defaults to False.
        """
        self.term_id = {}
        self.id_term = {}
        self.term_freq = {}
        self.ignorecase = ignorecase
        if terms is not None:
            self.add_terms(terms)

    def __repr__(self):
        """Returns a string representation showing the vocabulary size."""
        return f'{self.__class__.__name__}(vocabulary_size="{len(self.term_id)}")'

    def __len__(self):
        """Returns the number of distinct terms in the vocabulary."""
        return len(self.term_id)

    def __contains__(self, item):
        """Checks whether a term (str or Token) is in the vocabulary."""
        return self.has_term(item)

    def __iter__(self):
        """Iterates over the terms in the vocabulary."""
        for term in self.term_id:
            yield term

    def reset_index(self):
        """Clears all terms, identifiers and frequencies from the vocabulary."""
        self.term_id = {}
        self.id_term = {}
        self.term_freq = {}

    def add_terms(self, terms: Union[str, Token, List[Union[str, Token]]], reset_index: bool = False):
        """Add a list of terms to the vocabulary. Use 'reset_index=True' to reset
        the vocabulary before adding the terms.

        :param terms: a list of terms to add to the vocabulary
        :type terms: List[str]
        :param reset_index: a flag to indicate whether to empty the vocabulary before adding terms
        :type reset_index: bool
        """
        if reset_index is True:
            self.reset_index()
        if isinstance(terms, str) or isinstance(terms, Token):
            terms = [terms]
        for term in terms:
            if isinstance(term, Token):
                term = term.n
            term = term.lower() if self.ignorecase else term
            if term in self.term_id:
                continue
            self._index_term(term)

    def has_term(self, term: Union[str, Token], ignorecase: bool = None):
        """Checks whether a term is present in the vocabulary.

        Args:
            term (Union[str, Token]): The term (or Token, whose normalised string is used)
                to look up.
            ignorecase (bool, optional): Whether to lowercase the term before lookup.
                Defaults to the vocabulary's ``ignorecase`` setting.

        Returns:
            bool: True if the term is in the vocabulary, False otherwise.
        """
        if ignorecase is None:
            ignorecase = self.ignorecase
        if isinstance(term, Token):
            term = term.n
        term = term.lower() if ignorecase else term
        return True if term in self.term_id else False

    def _index_term(self, term: str):
        """Assigns the next available identifier to a new term."""
        term_id = len(self.term_id)
        self.term_id[term] = term_id
        self.id_term[term_id] = term

    def term2id(self, term: str):
        """Return the term ID for a given term.

        Args:
            term (str): The term to look up.

        Returns:
            Optional[int]: The term's identifier, or None if the term is not indexed.
        """
        return self.term_id[term] if term in self.term_id else None

    def id2term(self, term_id: int):
        """Return the term for a given term ID.

        Args:
            term_id (int): The identifier to look up.

        Returns:
            Optional[str]: The corresponding term, or None if the identifier is unknown.
        """
        return self.id_term[term_id] if term_id in self.id_term else None
