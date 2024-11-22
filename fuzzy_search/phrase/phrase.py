import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Union

from fuzzy_search.tokenization.string import SkipGram, text2skipgrams
from fuzzy_search.tokenization.token import Token
from fuzzy_search.tokenization.token import Tokenizer


def is_valid_label(label: Union[str, List[str]]) -> bool:
    """Test whether label has a valid value.

    :param label: a phrase label (either a string or a list of strings)
    :type label: Union[str, List[str]]
    :return: whether the label is valid
    :rtype: bool
    """
    if isinstance(label, list):
        for item in label:
            if not isinstance(item, str):
                return False
        return True
    return isinstance(label, str)


class Phrase:

    def __init__(self, phrase: Union[str, Dict[str, str]], ngram_size: int = 2, skip_size: int = 2,
                 early_threshold: int = 3, late_threshold: int = 3, within_range_threshold: int = 3,
                 ignorecase: bool = False, tokens: List[Token] = None, tokenizer: Tokenizer = None):
        if isinstance(phrase, str):
            phrase = {"phrase": phrase}
        self.name = phrase["phrase"]
        self.phrase_string = self.name if not ignorecase else self.name.lower()
        self.exact_string = re.escape(self.phrase_string)
        self.extact_word_boundary_string = re.compile(rf"\b{self.exact_string}\b")
        self.label = None
        self.max_start_offset: int = -1
        self.max_start_end: int = -1
        self.max_end_offset: int = -1
        self.max_end_start: int = -1
        self.label_set: Set[str] = set()
        self.label_list: List[str] = []
        self.properties = phrase
        self.ngram_size = ngram_size
        self.skip_size = skip_size
        self.early_threshold = early_threshold
        self.late_threshold = len(self.name) - late_threshold - ngram_size
        self.within_range_threshold = within_range_threshold
        self.ignorecase = ignorecase
        self.skipgrams = [skipgram for skipgram in text2skipgrams(self.phrase_string,
                                                                  ngram_size=ngram_size, skip_size=skip_size)]
        self.skipgram_set = set([skipgram.string for skipgram in self. skipgrams])
        self.skipgram_index: Dict[str, List[SkipGram]] = defaultdict(list)
        self.skipgram_index_lower: Dict[str, List[SkipGram]] = defaultdict(list)
        self.skipgram_freq = Counter([skipgram.string for skipgram in self.skipgrams])
        self.early_skipgram_index = {skipgram.string: skipgram for skipgram in
                                     self.skipgrams if skipgram.start_offset < early_threshold}
        self.late_skipgram_index = {skipgram.string: skipgram for skipgram in
                                    self.skipgrams if skipgram.start_offset + skipgram.length > self.late_threshold}
        # add lowercase version to allow both matching with and without ignorecase
        self.skipgrams_lower = [skipgram for skipgram in text2skipgrams(self.phrase_string.lower(),
                                                                        ngram_size=ngram_size, skip_size=skip_size)]
        self.early_skipgram_index_lower = {skipgram.string: skipgram for skipgram in self.skipgrams_lower
                                           if skipgram.start_offset < early_threshold}
        self.late_skipgram_index_lower = {skipgram.string: skipgram for skipgram in self.skipgrams_lower
                                          if skipgram.start_offset + skipgram.length > self.late_threshold}
        # print(self.late_skipgram_index_lower.keys())
        self.skipgram_freq_lower = Counter([skipgram.string for skipgram in self.skipgrams_lower])
        self.num_skipgrams = len(self.skipgrams)
        self.skipgram_distance = {}
        self.metadata: dict = phrase['metadata'] if 'metadata' in phrase else {}
        self.words: List[str] = [word for word in re.split(r"\W+", self.phrase_string) if word != ""]
        self.word_set: Set[str] = set(self.words)
        self.tokens: List[Token] = tokens
        self.token_index = defaultdict(list)
        self.first_word = None if len(self.words) == 0 else self.words[0]
        self.last_word = None if len(self.words) == 0 else self.words[-1]
        self.num_words = len(self.words)
        if "label" in phrase:
            self.set_label(phrase["label"])
        if len(phrase.keys()) > 1:
            self.add_metadata(phrase)
        self._index_skipgrams()
        self._set_within_range()
        if tokens is None and tokenizer is not None:
            self.tokens = tokenizer.tokenize(self.phrase_string)
            self.words = [token.string for token in self.tokens]
            for ti, token in enumerate(self.tokens):
                self.token_index[token.n].append(ti)

    def __repr__(self):
        return f"Phrase({self.phrase_string}, {self.label})"

    def __len__(self):
        return len(self.phrase_string)
    # internal methods

    def _index_skipgrams(self) -> None:
        """Turn the phrase into a list of skipgrams and index them with their offset(s) as values."""
        for skipgram in self.skipgrams:
            self.skipgram_index[skipgram.string] += [skipgram]
        for skipgram in self.skipgrams_lower:
            self.skipgram_index_lower[skipgram.string] += [skipgram]

    def _set_within_range(self):
        self.skipgram_distance = {}
        for index1 in range(0, len(self.skipgrams)-1):
            skipgram1 = self.skipgrams[index1]
            for index2 in range(index1+1, len(self.skipgrams)):
                skipgram2 = self.skipgrams[index2]
                if skipgram2.start_offset - skipgram1.start_offset > self.within_range_threshold:
                    continue
                if (skipgram1, skipgram2) not in self.skipgram_distance:
                    self.skipgram_distance[(skipgram1, skipgram2)] = skipgram2.start_offset - skipgram1.start_offset
                elif self.skipgram_distance[(skipgram1, skipgram2)] > skipgram2.start_offset - skipgram1.start_offset:
                    self.skipgram_distance[(skipgram1, skipgram2)] = skipgram2.start_offset - skipgram1.start_offset

    # external methods

    def set_label(self, label: Union[str, List[str]]) -> None:
        """Set the label(s) of a phrase. Labels must be string and can be a single string or a list.

        :param label: the label(s) of a phrase
        :type label: Union[str, List[str]]
        """
        if not is_valid_label(label):
            raise ValueError("phrase label must be a single string or a list of strings:", label)
        self.label = label
        if isinstance(label, str):
            self.label_set = {label}
            self.label_list = [label]
        else:
            self.label_set = set(label)
            self.label_list = label

    def has_label(self, label_string: str) -> bool:
        """Check if a given label belongs to at least one phrase in the phrase model.

        :param label_string: a label string
        :type label_string: str
        :return: a boolean whether the label is part of the phrase model
        :rtype: bool
        """
        if isinstance(self.label, list):
            return label_string in self.label
        else:
            return label_string == self.label

    def add_metadata(self, metadata_dict: Dict[str, any]) -> None:
        """Add key/value pairs as metadata for this phrase.

        :param metadata_dict: a dictionary of key/value pairs as metadata
        :type metadata_dict: Dict[str, any]
        :return: None
        :rtype: None
        """
        for key in metadata_dict:
            self.metadata[key] = metadata_dict[key]
            if key == "label":
                self.set_label(metadata_dict[key])
            elif key == "max_start_offset":
                self.add_max_start_offset(metadata_dict["max_start_offset"])
            elif key == "max_end_offset":
                self.add_max_end_offset(metadata_dict["max_end_offset"])

    def add_max_start_offset(self, max_start_offset: int) -> None:
        """Add a maximum offset from the start for matching a phrase in a text.

        :param max_start_offset: the maximum offset from the start to allow a phrase to match
        :type max_start_offset: int
        """
        if not isinstance(max_start_offset, int):
            raise TypeError("max_start_offset must be a positive integer")
        if max_start_offset < 0:
            raise ValueError("max_start_offset must be positive")
        self.max_start_offset = max_start_offset
        self.max_start_end = self.max_start_offset + len(self.phrase_string)

    def add_max_end_offset(self, max_end_offset: int) -> None:
        """Add a maximum offset from the end for matching a phrase in a text.

        :param max_end_offset: the maximum offset from the end to allow a phrase to match
        :type max_end_offset: int
        """
        if not isinstance(max_end_offset, int):
            raise TypeError("max_end_offset must be a positive integer")
        if max_end_offset < 0:
            raise ValueError("max_end_offset must be positive")
        self.max_end_offset = max_end_offset
        self.max_end_start = self.max_end_offset - len(self.phrase_string)

    def has_max_start_offset(self) -> bool:
        return self.max_start_offset is not None and self.max_start_offset >= 0

    def has_max_end_offset(self) -> bool:
        return self.max_end_offset is not None and self.max_end_offset >= 0

    def has_skipgram(self, skipgram: str) -> bool:
        """For a given skipgram, return boolean whether it is in the index

        :param skipgram: an skipgram string
        :type skipgram: str
        :return: A boolean whether skipgram is in the index
        :rtype: bool"""
        return skipgram in self.skipgram_index.keys()

    def skipgram_offsets(self, skipgram_string: str) -> Union[None, List[int]]:
        """For a given skipgram return the list of offsets at which it appears.

        :param skipgram_string: an skipgram string
        :type skipgram_string: str
        :return: A list of string offsets at which the skipgram appears
        :rtype: Union[None, List[int]]"""
        if not self.has_skipgram(skipgram_string):
            return None
        return [skipgram.start_offset for skipgram in self.skipgram_index[skipgram_string]]

    def within_range(self, skipgram1, skipgram2):
        if not self.has_skipgram(skipgram1) or not self.has_skipgram(skipgram2):
            return False
        elif (skipgram1, skipgram2) not in self.skipgram_distance:
            return False
        elif self.skipgram_distance[(skipgram1, skipgram2)] > self.within_range_threshold:
            return False
        else:
            return True

    def is_early_skipgram(self, skipgram: str) -> bool:
        """For a given skipgram, return boolean whether it appears early in the phrase.

        :param skipgram: an skipgram string
        :type skipgram: str
        :return: A boolean whether skipgram appears early in the phrase
        :rtype: bool"""
        return skipgram in self.early_skipgram_index
