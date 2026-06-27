"""Candidate match objects and the skipgram-based logic used to build, grow and validate
fuzzy match candidates between a text and a phrase."""

from __future__ import annotations
from collections import Counter
from typing import Dict, List, Union

from fuzzy_search.tokenization.string import SkipGram
from fuzzy_search.phrase.phrase import Phrase


class Candidate:
    """A finalized candidate match between a phrase and a span of text, with the matching
    string, its offsets, and the skipgram overlap score."""

    def __init__(self, phrase: Phrase, match_start_offset: int,
                 match_end_offset: int, match_string: str, skipgram_overlap: float = 0.0):
        """Create a Candidate for a matched phrase.

        :param phrase: the phrase that this candidate is a match for
        :type phrase: Phrase
        :param match_start_offset: the start offset of the match in the text
        :type match_start_offset: int
        :param match_end_offset: the end offset of the match in the text
        :type match_end_offset: int
        :param match_string: the matching text string
        :type match_string: str
        :param skipgram_overlap: the skipgram overlap score between phrase and match string
        :type skipgram_overlap: float
        """
        self.phrase = phrase
        self.match_start_offset: int = match_start_offset
        self.match_end_offset: int = match_end_offset
        self.match_string: str = match_string
        self.skipgram_overlap: float = skipgram_overlap

    def __repr__(self):
        """Return a debug representation showing the phrase, match string, and offsets."""
        return f'Candidate(' + \
               f'phrase: "{self.phrase.phrase_string}", match_string: "{self.match_string}", ' + \
               f'match_start_offset: {self.match_start_offset}, match_end_offset: {self.match_end_offset})'


class CandidatePartial:
    """A partially built candidate match for a phrase, accumulating matching skipgrams found in
    the text as they are encountered. Used while scanning text for skipgram matches, before a
    final Candidate is produced."""

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
        """Return a debug representation showing the phrase, match string, and offsets."""
        return f'Candidate(' + \
               f'phrase: "{self.phrase.phrase_string}", match_string: "{self.match_string}", ' + \
               f'match_start_offset: {self.match_start_offset}, match_end_offset: {self.match_end_offset})'


def candidate_from_partial(candidate_partial: CandidatePartial, text: Dict[str, any]) -> Candidate:
    """Create a finalized Candidate instance from a CandidatePartial object.

    :param candidate_partial: a partially built candidate match
    :type candidate_partial: CandidatePartial
    :param text: the text object that the candidate match string is taken from
    :type text: Dict[str, any]
    :return: a finalized Candidate
    :rtype: Candidate
    """
    if candidate_partial.match_string is None:
        match_string = get_match_string(candidate_partial, text)
    else:
        match_string = candidate_partial.match_string
    return Candidate(candidate_partial.phrase, candidate_partial.match_start_offset,
                     candidate_partial.match_end_offset, match_string,
                     get_skip_count_overlap(candidate_partial))


def same_candidate(candidate1: Union[CandidatePartial, Candidate],
                   candidate2: Union[CandidatePartial, Candidate]):
    """Check if this candidate has the same start and end offsets as another candidate.

    :param candidate1: first candidate .
    :type candidate1: Candidate
    :param candidate2: second candidate.
    :type candidate2: Candidate
    :return: this candidate match has the same offsets as the other candidate
    :rtype: bool
    """
    if candidate1.match_start_offset != candidate2.match_start_offset:
        return False
    if candidate1.match_end_offset != candidate2.match_end_offset:
        return False
    else:
        return True


def add_skip_match(candidate: CandidatePartial, skipgram: SkipGram) -> None:
    """Add a skipgram match between a text and a phrase to the candidate, updating its offsets
    and skipgram counts. If adding the skipgram makes the candidate's matched span longer than
    the phrase allows, or its start no longer lies in the phrase's early skipgram index, the
    earliest skipgrams are dropped from the front until the candidate is valid again.

    :param candidate: the candidate to add the skipgram to
    :type candidate: CandidatePartial
    :param skipgram: a matching skipgram
    :type skipgram: SkipGram
    """
    if len(candidate.skipgram_list) == 0 and skipgram.string not in candidate.early_skipgram_index:
        if candidate.debug > 3:
            print("skipping skipgram as first for candidate:", skipgram.string)
        return None
    candidate.skipgram_set.add(skipgram.string)
    candidate.skipgram_list.append(skipgram)
    if candidate.match_start_offset is None or candidate.match_start_offset < 0:
        candidate.match_start_offset = get_match_start_offset(candidate)
    if skipgram.start_offset + skipgram.length > candidate.match_end_offset:
        candidate.match_end_offset = skipgram.start_offset + skipgram.length
    candidate.skipgram_count.update([skipgram.string])
    if candidate.debug > 2:
        print("\tadd - skipgram:", skipgram.string, skipgram.start_offset)
        print("\tadd - match length:", get_skip_match_length(candidate))
        print("\tadd - list:", [skip.string for skip in candidate.skipgram_list])
    # check if the candidate string is too long to match the phrase
    # if too long, remove the first skipgrams until the string is short enough
    while get_skip_match_length(candidate) > candidate.max_length and len(candidate.skipgram_list) > 0:
        remove_first_skip(candidate)
        candidate.match_start_offset = get_match_start_offset(candidate)
        if candidate.debug > 2:
            print("\tremove - too long - length:", get_skip_match_length(candidate))
            print("\tremove - too long - list:", [skip.string for skip in candidate.skipgram_list])
            print("\tremove - too long - start:", candidate.match_start_offset, "\tend:", candidate.match_end_offset)
    while len(candidate.skipgram_list) > 0 and candidate.skipgram_list[0].string not in candidate.early_skipgram_index:
        remove_first_skip(candidate)
        candidate.match_start_offset = get_match_start_offset(candidate)
        if candidate.debug > 2:
            print("\tremove - no start - length:", get_skip_match_length(candidate))
            print("\tremove - no start - list:", [skip.string for skip in candidate.skipgram_list])
            print("\tremove - no start - start:", candidate.match_start_offset, "\tend:", candidate.match_end_offset)

def shift_start_skip(candidate: CandidatePartial) -> bool:
    """Check if a later skipgram in the candidate would make a better starting point than the
    current first skipgram (i.e. it shortens an overly long match without losing the best start),
    and if so, shift the candidate's start to that skipgram.

    :param candidate: a candidate match to check and potentially shift
    :type candidate: CandidatePartial
    :return: whether the start was shifted
    :rtype: bool
    """
    if get_skip_match_length(candidate) <= len(candidate.phrase.phrase_string):
        return False
    start_skip = candidate.skipgram_list[0]
    start_phrase_offset = candidate.skipgram_index[start_skip.string][0].start_offset
    best_start_phrase_offset = start_phrase_offset
    best_start_index = 0
    best_start_skip = start_skip
    for si, skip in enumerate(candidate.skipgram_list):
        skip_phrase_offset = candidate.skipgram_index[skip.string][0].start_offset
        if skip.start_offset - start_skip.start_offset > get_skip_match_length(candidate) - len(candidate.phrase.phrase_string):
            # stop looking for better start when remaining skips result in too short match length
            break
        if skip.start_offset > best_start_skip.start_offset and skip_phrase_offset <= best_start_phrase_offset:
            best_start_index = si
            best_start_skip = skip
            best_start_phrase_offset = skip_phrase_offset
        if skip.string not in candidate.early_skipgram_index:
            break
    for _ in range(0, best_start_index):
        remove_first_skip(candidate)
    candidate.match_start_offset = get_match_start_offset(candidate)
    return best_start_index > 0

def remove_first_skip(candidate: CandidatePartial) -> None:
    """Remove the first matching skipgram from the candidate's list and update its skipgram
    count and set accordingly.

    :param candidate: the candidate to remove the first skipgram from
    :type candidate: CandidatePartial
    """
    first_skip = candidate.skipgram_list.pop(0)
    if candidate.debug > 3:
        print('\tremove_first_skip - removing first skip')
    # reduce count of first skipgram by 1
    candidate.skipgram_count[first_skip.string] -= 1
    # if count has dropped to zero, remove skipgram from the set
    if candidate.skipgram_count[first_skip.string] == 0:
        candidate.skipgram_set.remove(first_skip.string)

def get_skip_match_length(candidate: CandidatePartial) -> int:
    """Return the length of the matching string spanned by the candidate's current offsets.

    :param candidate: the candidate to measure
    :type candidate: CandidatePartial
    :return: the length of the candidate's matching span
    :rtype: int
    """
    if candidate.match_start_offset is None:
        return 0
    return candidate.match_end_offset - candidate.match_start_offset

def is_match(candidate: CandidatePartial, skipgram_threshold: float) -> bool:
    """Check if the candidate is a likely match for its corresponding phrase, based on its
    length relative to the phrase, whether its first/last skipgrams are in the phrase's
    early/late skipgram indexes, and whether its skipgram set overlap meets the given threshold.

    :param candidate: the candidate to validate
    :type candidate: CandidatePartial
    :param skipgram_threshold: the minimum required skipgram set overlap
    :type skipgram_threshold: float
    :return: whether the candidate is a likely match
    :rtype: bool
    """
    if len(candidate.skipgram_list) == 0:
        if candidate.debug > 3:
            print('\tis_match - NO MATCH: there are no matching skipgrams')
        return False
    if candidate.skipgram_list[0].string not in candidate.early_skipgram_index:
        if candidate.debug > 3:
            print('\tis_match - NO MATCH: first skipgram not in early index')
        return False
    
    phrase_len = len(candidate.phrase.phrase_string)
    match_len = get_skip_match_length(candidate)
    
    if match_len > phrase_len + candidate.max_length_variance:
        if candidate.debug > 3:
            print('\tis_match - NO MATCH: skip match length too long')
        return False
    elif match_len < candidate.phrase.late_threshold - candidate.max_length_variance:
        return False

    if candidate.skipgram_list[-1].string not in candidate.late_skipgram_index:
        return False
    if get_skip_set_overlap(candidate) < skipgram_threshold:
        return False
    return True

def get_skip_set_overlap(candidate: CandidatePartial) -> float:
    """Calculate the skipgram set overlap between the candidate and its phrase (fraction of the
    phrase's distinct skipgrams that are present in the candidate), store it on the candidate,
    and return it.

    :param candidate: the candidate to score
    :type candidate: CandidatePartial
    :return: the skipgram set overlap
    :rtype: float
    """
    candidate.skipgram_overlap = len(candidate.skipgram_set) / len(candidate.phrase.skipgram_set)
    return candidate.skipgram_overlap


def get_skip_count_overlap(candidate: CandidatePartial) -> float:
    """Calculate deviation of candidate skipgrams from phrase skipgrams.

            :param candidate: the candidate to score
            :type candidate: CandidatePartial
            :return: the skipgram overlap (-inf, 1.0]
            :rtype: float
            """
    diff = 0
    total = 0
    for skipgram_string, count in candidate.skipgram_count.items():
        diff += abs(count - candidate.skipgram_freq[skipgram_string])
        total += count
    return (total - diff) / candidate.phrase.num_skipgrams

def get_match_start_offset(candidate: CandidatePartial) -> Union[None, int]:
    """Calculate the start offset of the match in the text, based on the candidate's first
    skipgram and that skipgram's offset within the phrase.

    :param candidate: the candidate to compute the start offset for
    :type candidate: CandidatePartial
    :return: the match start offset, or None if the candidate has no skipgrams
    :rtype: Union[None, int]
    """
    if len(candidate.skipgram_list) == 0:
        return None
    first_skip = candidate.skipgram_list[0]
    first_skip_in_phrase = candidate.skipgram_index[first_skip.string][0]
    match_start_offset = candidate.skipgram_list[0].start_offset - first_skip_in_phrase.start_offset
    return 0 if match_start_offset < 0 else match_start_offset

def get_match_string(candidate: CandidatePartial, text: Dict[str, any]) -> Union[str, None]:
    """Find the matching string of a candidate fuzzy match by slicing the text between the
    candidate's start and end offsets.

    :param candidate: the candidate whose match string to extract
    :type candidate: CandidatePartial
    :param text: the text object containing the candidate's match string
    :type text: Dict[str, any]
    :return: the matching text string
    :rtype: Union[str, None]
    """
    if candidate.match_start_offset == candidate.match_end_offset:
        raise ValueError('start and end offset cannot be the same')
    return text["text"][candidate.match_start_offset:candidate.match_end_offset]
