from typing import List, Union

from .token import Token


class Vocabulary:

    def __init__(self, terms: List[Union[str, Token]] = None, ignorecase: bool = False):
        """A Vocabulary class to map terms to identifiers."""
        self.term_id = {}
        self.id_term = {}
        self.term_freq = {}
        self.ignorecase = ignorecase
        if terms is not None:
            self.add_terms(terms)

    def __repr__(self):
        return f'{self.__class__.__name__}(vocabulary_size="{len(self.term_id)}")'

    def __len__(self):
        return len(self.term_id)

    def __contains__(self, item):
        return self.has_term(item)

    def __iter__(self):
        for term in self.term_id:
            yield term

    def reset_index(self):
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
        if ignorecase is None:
            ignorecase = self.ignorecase
        if isinstance(term, Token):
            term = term.n
        term = term.lower() if ignorecase else term
        return True if term in self.term_id else False

    def _index_term(self, term: str):
        term_id = len(self.term_id)
        self.term_id[term] = term_id
        self.id_term[term_id] = term

    def term2id(self, term: str):
        """Return the term ID for a given term."""
        return self.term_id[term] if term in self.term_id else None

    def id2term(self, term_id: int):
        """Return the term for a given term ID."""
        return self.id_term[term_id] if term_id in self.id_term else None
