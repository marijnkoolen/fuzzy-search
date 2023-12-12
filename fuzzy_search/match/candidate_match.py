from __future__ import annotations
from collections import Counter
from typing import Dict, List, Union

from fuzzy_search.tokenization.string import SkipGram
from fuzzy_search.phrase.phrase import Phrase


class Candidate:

    def __init__(self, phrase: Phrase, max_length_variance: int = 1,
                 ignorecase: bool = False, debug: int = 0):
        """Create a Candidate instance for a given Phrase object.

        :param phrase: a phrase object
        :type phrase: Phrase
        :param ignorecase: whether to ignore case when matching skip grams
        :type ignorecase: bool
        :param debug: level to show debugging info
        :type debug: int
        """
        self.skipgram_set = set()
        self.skipgram_list: List[SkipGram] = []
        self.skipgram_count = Counter()
        self.phrase = phrase
        self.ignorecase = ignorecase
        self.debug = debug
        if ignorecase:
            self.skipgrams = phrase.skipgrams_lower
            self.skipgram_index = phrase.skipgram_index_lower
            self.skipgram_freq = phrase.skipgram_freq_lower
            self.early_skipgram_index = phrase.early_skipgram_index_lower
            self.late_skipgram_index = phrase.late_skipgram_index_lower
        else:
            self.skipgrams = phrase.skipgrams
            self.skipgram_index = phrase.skipgram_index
            self.skipgram_freq = phrase.skipgram_freq
            self.early_skipgram_index = phrase.early_skipgram_index
            self.late_skipgram_index = phrase.late_skipgram_index
        self.max_length_variance = max_length_variance
        self.max_length = len(self.phrase.phrase_string) + self.max_length_variance
        self.match_start_offset: int = -1
        self.match_end_offset: int = -1
        self.match_string: Union[None, str] = None
        self.skipgram_overlap: float = 0.0

    def __repr__(self):
        return f'Candidate(' + \
               f'phrase: "{self.phrase.phrase_string}", match_string: "{self.match_string}", ' + \
               f'match_start_offset: {self.match_start_offset}, match_end_offset: {self.match_end_offset})'

    def add_skip_match(self, skipgram: SkipGram) -> None:
        """Add a skipgram match between a text and a phrase ot the candidate.

        :param skipgram: a matching skipgram
        :type skipgram: SkipGram
        """
        if len(self.skipgram_list) == 0 and skipgram.string not in self.early_skipgram_index:
            if self.debug > 2:
                print("skipping skipgram as first for candidate:", skipgram.string)
            return None
        self.skipgram_set.add(skipgram.string)
        self.skipgram_list.append(skipgram)
        if self.match_start_offset is None or self.match_start_offset < 0:
            self.match_start_offset = self.get_match_start_offset()
        if skipgram.start_offset + skipgram.length > self.match_end_offset:
            self.match_end_offset = skipgram.start_offset + skipgram.length
        self.skipgram_count.update([skipgram.string])
        if self.debug > 2:
            print("\tadd - skipgram:", skipgram.string, skipgram.start_offset)
            print("\tadd - match length:", self.skip_match_length())
            print("\tadd - list:", [skip.string for skip in self.skipgram_list])
        # check if the candidate string is too long to match the phrase
        # if too long, remove the first skipgrams until the string is short enough
        while self.skip_match_length() > self.max_length and len(self.skipgram_list) > 0:
            self.remove_first_skip()
            self.match_start_offset = self.get_match_start_offset()
            if self.debug > 2:
                print("\tremove - too long - length:", self.skip_match_length())
                print("\tremove - too long - list:", [skip.string for skip in self.skipgram_list])
                print("\tremove - too long - start:", self.match_start_offset, "\tend:", self.match_end_offset)
        while len(self.skipgram_list) > 0 and self.skipgram_list[0].string not in self.early_skipgram_index:
            self.remove_first_skip()
            self.match_start_offset = self.get_match_start_offset()
            if self.debug > 2:
                print("\tremove - no start - length:", self.skip_match_length())
                print("\tremove - no start - list:", [skip.string for skip in self.skipgram_list])
                print("\tremove - no start - start:", self.match_start_offset, "\tend:", self.match_end_offset)

    def shift_start_skip(self) -> bool:
        """Check if there is a later skip that is a better start."""
        if self.skip_match_length() <= len(self.phrase.phrase_string):
            return False
        start_skip = self.skipgram_list[0]
        start_phrase_offset = self.skipgram_index[start_skip.string][0].start_offset
        best_start_phrase_offset = start_phrase_offset
        best_start_index = 0
        best_start_skip = start_skip
        for si, skip in enumerate(self.skipgram_list):
            skip_phrase_offset = self.skipgram_index[skip.string][0].start_offset
            if skip.start_offset - start_skip.start_offset > self.skip_match_length() - len(self.phrase.phrase_string):
                # stop looking for better start when remaining skips result in too short match length
                break
            if skip.start_offset > best_start_skip.start_offset and skip_phrase_offset <= best_start_phrase_offset:
                best_start_index = si
                best_start_skip = skip
                best_start_phrase_offset = skip_phrase_offset
            if skip.string not in self.early_skipgram_index:
                break
        for _ in range(0, best_start_index):
            self.remove_first_skip()
        self.match_start_offset = self.get_match_start_offset()
        return best_start_index > 0

    def remove_first_skip(self) -> None:
        """Remove the first matching skipgram from the list and update the count and set."""
        first_skip = self.skipgram_list.pop(0)
        if self.debug > 3:
            print('\tremove_first_skip - removing first skip')
        # reduce count of first skipgram by 1
        self.skipgram_count[first_skip.string] -= 1
        # if count has dropped to zero, remove skipgram from the set
        if self.skipgram_count[first_skip.string] == 0:
            self.skipgram_set.remove(first_skip.string)

    def skip_match_length(self) -> int:
        """Return the length of the matching string.

        :return: difference between start and end offset
        :rtype: int
        """
        if self.match_start_offset is None:
            return 0
        return self.match_end_offset - self.match_start_offset

    def is_match(self, skipgram_threshold: float):
        """Check if the candidate is a likely match for its corresponding phrase.

        :param skipgram_threshold: the threshold to for how many skipgrams have to match between candidate and phrase
        :type skipgram_threshold: float
        :return: a boolean whether this candidate is a likely match for the phrase
        :rtype: bool
        """
        if len(self.skipgram_list) == 0:
            # there are no matching skipgrams, so no matching string
            if self.debug > 2:
                print('\tis_match - NO MATCH: there are no matching skipgrams, so no matching string')
            return False
        if self.skipgram_list[0].string not in self.early_skipgram_index:
            # the first skipgram of candidate is not in the early skipgrams of phrase
            if self.debug > 2:
                print('\tis_match - NO MATCH: the first skipgram of candidate is not in the early skipgrams of phrase')
            return False
        # length of current skip matches should be no longer than phrase plus max length variance
        # but also no shorter than phrase minus max length variance minus late skips offset from end
        # of phrase.
        if self.skip_match_length() > len(self.phrase.phrase_string) + self.max_length_variance:
            if self.debug > 2:
                print('\tis_match - NO MATCH: skip match length is longer than max length variance of phrase')
            return False
        elif self.skip_match_length() < self.phrase.late_threshold - self.max_length_variance:
            if self.debug > 3:
                print('\tis_match - phrase length:', len(self.phrase.phrase_string))
                print('\tis_match - skip match length:', self.skip_match_length())
                print('\tis_match - max length variance:', self.max_length_variance)
                print('\tis_match - late threshold:', self.phrase.late_threshold)
                print('\tis_match - NO MATCH: skip match length is not within max length variance of phrase')
            return False
        if self.debug > 3:
            print('\tis_match - candidate phrase:', self.phrase.phrase_string)
            print('\tis_match - skip_set_overlap:', self.get_skip_set_overlap())
            print('\tis_match - late threshold:', self.phrase.late_threshold)
        if self.skipgram_list[-1].string not in self.late_skipgram_index:
            if self.debug > 2:
                print("\tis_match - NO MATCH: last skip not in late index")
            # the last skipgram of candidate is not in the late skipgrams of phrase
            return False
        if self.get_skip_set_overlap() < skipgram_threshold:
            if self.debug > 2:
                print("\tis_match - below skipgram threshold:", self.get_skip_set_overlap(), skipgram_threshold)
            return False
        else:
            if self.debug > 2:
                print('\tis_match - MATCH!')
            return True

    def get_skip_set_overlap(self) -> float:
        """Calculate and set skipgram overlap between text and phrase skipgram matches.

        :return: the skipgram overlap
        :rtype: float
        """
        self.skipgram_overlap = len(self.skipgram_set) / len(self.phrase.skipgram_set)
        return self.skipgram_overlap

    def get_skip_count_overlap(self) -> float:
        """Calculate deviation of candidate skipgrams from phrase skipgrams.

        :return: the skipgram overlap (-inf, 1.0]
        :rtype: float
        """
        diff = 0
        total = 0
        for skipgram_string, count in self.skipgram_count.items():
            diff += abs(count - self.skipgram_freq[skipgram_string])
            total += count
        return (total - diff) / self.phrase.num_skipgrams

    def get_match_start_offset(self) -> Union[None, int]:
        """Calculate the start offset of the match.

        :return: the start offset of the match
        :rtype: int
        """
        if len(self.skipgram_list) == 0:
            return None
        first_skip = self.skipgram_list[0]
        first_skip_in_phrase = self.skipgram_index[first_skip.string][0]
        match_start_offset = self.skipgram_list[0].start_offset - first_skip_in_phrase.start_offset
        if self.debug > 3:
            print("\tget_match_start_offset - in match:", first_skip.string, first_skip.start_offset)
            print("\tget_match_start_offset - in phrase:",
                  first_skip_in_phrase.string, first_skip_in_phrase.start_offset)
            print("\tget_match_start_offset - match_start_offset:", match_start_offset)
        return 0 if match_start_offset < 0 else match_start_offset

    def get_match_string(self, text: Dict[str, any]) -> Union[str, None]:
        """Find the matching string of a candidate fuzzy match between a text and a phrase.

        :param text: the text object from which the candidate was derived
        :type text: Dict[str, any]
        :return: the matching string
        :rtype: str
        """
        if self.debug > 2:
            print('\tget_match_string - text:', text)
        if self.match_start_offset == self.match_end_offset:
            raise ValueError('start and end offset cannot be the same')
        if self.match_start_offset > self.match_end_offset:
            raise ValueError('start offset cannot be bigger than end offset')
        return text["text"][self.match_start_offset:self.match_end_offset]
    # TODO: check if first to last offset is too long
    # if not, the match string is probably fine
    # if it is, find the best substring

    def same_candidate(self, other: Candidate):
        """Check if this candidate has the same start and end offsets as another candidate.

        :param other: another candidate for the same phrase and text.
        :type other: Candidate
        :return: this candidate match has the same offsets as the other candidate
        :rtype: bool
        """
        if self.match_start_offset != other.match_start_offset:
            return False
        if self.match_end_offset != other.match_end_offset:
            return False
        else:
            return True

