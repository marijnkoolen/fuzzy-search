from typing import Dict, List, Union
from collections import defaultdict, Counter

from fuzzy_search.fuzzy_string import SkipGram, text2skipgrams


class Phrase(object):

    def __init__(self, phrase: Union[str, Dict[str, str]], ngram_size: int = 2, skip_size: int = 2,
                 early_threshold: int = 3, late_threshold: int = 3, within_range_threshold: int = 3,
                 ignore_case: bool = False):
        if isinstance(phrase, str):
            phrase = {"phrase": phrase}
        self.name = phrase["phrase"]
        self.phrase_string = self.name if not ignore_case else self.name.lower()
        self.properties = phrase
        self.ngram_size = ngram_size
        self.skip_size = skip_size
        self.early_threshold = early_threshold
        self.late_threshold = len(self.name) - late_threshold - ngram_size
        self.within_range_threshold = within_range_threshold
        self.ignore_case = ignore_case
        self.skipgrams = [skipgram for skipgram in text2skipgrams(self.phrase_string,
                                                                  ngram_size=ngram_size, skip_size=skip_size)]
        self.skipgram_set = set([skipgram.string for skipgram in self. skipgrams])
        self.skipgram_index: Dict[str, List[SkipGram]] = defaultdict(list)
        self.skipgram_index_lower: Dict[str, List[SkipGram]] = defaultdict(list)
        self.skipgram_freq = Counter([skipgram.string for skipgram in self.skipgrams])
        self.early_skipgram_index = {skipgram.string: skipgram for skipgram in
                                     self.skipgrams if skipgram.offset < early_threshold}
        self.late_skipgram_index = {skipgram.string: skipgram for skipgram in
                                    self.skipgrams if skipgram.offset > self.late_threshold}
        # add lowercase version to allow both matching with and without ignore_case
        self.skipgrams_lower = [skipgram for skipgram in text2skipgrams(self.phrase_string.lower(),
                                                                        ngram_size=ngram_size, skip_size=skip_size)]
        self.early_skipgram_index_lower = {skipgram: skipgram for skipgram in self.skipgrams_lower
                                           if skipgram.offset < early_threshold}
        self.late_skipgram_index_lower = {skipgram.string: skipgram for skipgram in self.skipgrams_lower
                                          if skipgram.offset > self.late_threshold}
        self.skipgram_freq_lower = Counter([skipgram.string for skipgram in self.skipgrams_lower])
        self.num_skipgrams = len(self.skipgrams)
        self.skipgram_distance = {}
        self.metadata: dict = phrase
        self._index_skipgrams()
        self._set_within_range()

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
                if skipgram2.offset - skipgram1.offset > self.within_range_threshold:
                    continue
                if (skipgram1, skipgram2) not in self.skipgram_distance:
                    self.skipgram_distance[(skipgram1, skipgram2)] = skipgram2.offset - skipgram1.offset
                elif self.skipgram_distance[(skipgram1, skipgram2)] > skipgram2.offset - skipgram1.offset:
                    self.skipgram_distance[(skipgram1, skipgram2)] = skipgram2.offset - skipgram1.offset

    # external methods

    def add_metadata(self, metadata_dict: Dict[str, any]) -> None:
        """Add key/value pairs as metadata for this phrase.

        :param metadata_dict: a dictionary of key/value pairs as metadata
        :type metadata_dict: Dict[str, any]
        :return: None
        :rtype: None
        """
        for key in metadata_dict:
            self.metadata[key] = metadata_dict[key]

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
        return [skipgram.offset for skipgram in self.skipgram_index[skipgram_string]]

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
