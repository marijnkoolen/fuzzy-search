import copy
from collections import defaultdict
from typing import Dict, List, Set, Union

from fuzzy_search.match.phrase_match import Candidate
from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.phrase.phrase_model import PhraseModel
from fuzzy_search.tokenization.string import SkipGram
from fuzzy_search.tokenization.string import score_levenshtein_similarity_ratio
from fuzzy_search.tokenization.token import Token


class SkipMatches:

    def __init__(self, ngram_size: int, skip_size: int):
        self.ngram_size = ngram_size
        self.skip_size = skip_size
        self.skip_length = ngram_size + skip_size
        self.match_set: Dict[Union[Phrase, Token, str], set] = defaultdict(set)
        self.match_offsets = defaultdict(list)
        self.match_skipgrams: Dict[Union[Phrase, Token, str], List[SkipGram]] = defaultdict(list)
        self.matches: Set[Union[Phrase, Token, str]] = set()

    def add_skip_match(self, skipgram: SkipGram, phrase: Union[Phrase, Token]) -> None:
        """Add a skipgram from a text that matches a phrase.

        :param skipgram: a skipgram from a text
        :type skipgram: SkipGram
        :param phrase: a phrase object that matches the skipgram
        :type phrase: Phrase
        """
        # track which skip grams of phrase are found in text
        self.match_set[phrase].add(skipgram.string)
        # track which text offsets match phrase skipgram
        self.match_offsets[phrase].append(skipgram.offset)
        self.match_skipgrams[phrase].append(skipgram)
        self.matches.add(phrase)


def get_skipset_overlap(phrase: Phrase, skip_matches: SkipMatches) -> float:
    """Calculate the overlap between the set of skipgrams of a text and the skipgrams of a phrase.

    :param phrase: a phrase object that has been matched against a text
    :type phrase: Phrase
    :param skip_matches: a SkipMatches object containing the skipgram matches between a text and a list of phrases
    :type skip_matches: SkipMatches
    :return: the fraction of skipgrams in the phrase that overlaps with the text
    :rtype: float
    """
    return len(skip_matches.match_set[phrase]) / len(phrase.skipgram_set)


def filter_skipgram_threshold(skip_matches: SkipMatches, skip_threshold: float) -> List[Phrase]:
    """Filter the skipgram matches based on the skipgram overlap threshold.

    :param skip_matches: the phrases that matches the text
    :type skip_matches: SkipMatches
    :param skip_threshold: the threshold for the skipgram overlap between a text and a phrase
    :type skip_threshold: float
    :return: the list of phrases with a skipgram overlap that meets the threshold
    :rtype: List[Phrase]
    """
    return [phrase for phrase in skip_matches.matches if get_skipset_overlap(phrase, skip_matches) >= skip_threshold]


def filter_overlapping_phrase_candidates(phrase_candidates: List[Candidate]) -> List[Candidate]:
    filtered: List[Candidate] = []
    if len(phrase_candidates) < 2:
        return phrase_candidates
    phrase_candidates.sort(key=lambda x: x.match_start_offset)
    prev_candidate = phrase_candidates[0]
    prev_score = score_levenshtein_similarity_ratio(prev_candidate.phrase.phrase_string, prev_candidate.match_string)
    for ci, curr_candidate in enumerate(phrase_candidates[1:]):
        if curr_candidate.match_end_offset > prev_candidate.match_start_offset:
            if curr_candidate.match_start_offset < prev_candidate.match_end_offset:
                # this candidate overlaps with the previous one, pick the best
                curr_score = score_levenshtein_similarity_ratio(curr_candidate.phrase.phrase_string,
                                                                curr_candidate.match_string)
                if curr_score > prev_score:
                    # this candidate is better, so skip the previous candidate
                    prev_candidate = curr_candidate
                    prev_score = curr_score
                elif curr_score == prev_score and len(curr_candidate.match_string) > len(prev_candidate.match_string):
                    # this candidate is longer with the same score, so skip the previous candidate
                    prev_candidate = curr_candidate
                    prev_score = curr_score
            else:
                # the previous candidate does not overlap with the current, so add it to filtered
                filtered.append(prev_candidate)
                prev_candidate = curr_candidate
                prev_score = score_levenshtein_similarity_ratio(curr_candidate.phrase.phrase_string,
                                                                curr_candidate.match_string)

    if len(filtered) == 0 or prev_candidate != filtered[-1]:
        filtered.append(prev_candidate)
    return filtered


def get_skipmatch_phrase_candidates(text: Dict[str, any], phrase: Phrase, skip_matches: SkipMatches,
                                    skipgram_threshold: float, max_length_variance: int = 1,
                                    ignorecase: bool = False, debug: int = 0) -> List[Candidate]:
    """Find all candidate matches for a given phrase and SkipMatches object.

    :param text: the text object to match with phrases
    :type text: Dict[str, any]
    :param phrase: a phrase to find candidate matches for
    :type phrase: Phrase
    :param skip_matches: a Skipmatches object with matches between a text and a list of phrases
    :type skip_matches: SkipMatches
    :param skipgram_threshold: a threshold for how many skipgrams should match between a phrase and a candidate
    :type skipgram_threshold: float
    :param max_length_variance: the maximum difference in length between candidate and phrase
    :type max_length_variance: int
    :param ignorecase: whether to ignore case when matching skip grams
    :type ignorecase: bool
    :param debug: level to show debug information
    :type debug: int
    :return: a list of candidate matches
    :rtype: List[Candidate]
    """
    candidates: List[Candidate] = []
    candidate = Candidate(phrase, max_length_variance=max_length_variance, ignorecase=ignorecase, debug=debug)
    last_index = len(skip_matches.match_offsets[phrase]) - 1
    if debug > 1:
        print(f"get_skipmatch_phrase_candidates - finding candidates for phrase ({len(phrase.phrase_string)}):", phrase.phrase_string)
        print('\t', skip_matches.match_offsets[phrase])
    for ci, curr_offset in enumerate(skip_matches.match_offsets[phrase]):
        next_offset = None if ci == last_index else skip_matches.match_offsets[phrase][ci + 1]
        if debug > 1:
            print('\t', ci, 'curr offset:', curr_offset, '\tskip:',
                  skip_matches.match_skipgrams[phrase][ci].string, '\tnext offset:', next_offset)
        # add current skipgram to the candidate
        candidate.add_skip_match(skip_matches.match_skipgrams[phrase][ci])
        if debug > 1 and abs(candidate.skip_match_length() - len(candidate.phrase.phrase_string)) < max_length_variance:
            skip = skip_matches.match_skipgrams[phrase][ci]
            print('\t', ci, curr_offset, "adding skip match:", skip.string, skip.offset, skip.length)
            print("\tcandidate skips:", [skip.string for skip in candidate.skipgram_list],
                  candidate.skip_match_length())
            print(candidate.get_skip_set_overlap(), candidate.get_match_string(text))
        # check if the current candidate is a potential match for the phrase
        if candidate.is_match(skipgram_threshold):
            candidate.match_string = candidate.get_match_string(text)
            if debug > 0:
                print("\tmeets threshold:", candidate.match_string)
            # if this candidate has enough skipgram overlap, yield it as a candidate match
            if len(candidates) == 0 or not candidate.same_candidate(candidates[-1]):
                candidates.append(copy.deepcopy(candidate))
            if candidate.shift_start_skip():
                # candidate string is longer than phrase string check if shifting the start creates
                # a better candidate and if so, add that as well
                candidate.match_string = candidate.get_match_string(text)
                candidates.append(copy.deepcopy(candidate))
        if next_offset and next_offset - curr_offset > skip_matches.ngram_size + skip_matches.skip_size + 1:
            # if the gap between the current skipgram and the next is larger than an entire skipgram
            # the next skipgram does not belong to this candidate
            # start a new candidate for the next skipgram
            if debug > 1:
                print('\tcurr_offset:', curr_offset, '\tnext_offset:', next_offset)
                print('\tstarting a new candidate')
            candidate = Candidate(phrase, max_length_variance=max_length_variance,
                                  ignorecase=ignorecase, debug=debug)
    # end of skipgrams reached, check if remaining candidate is a match
    if debug > 1:
        print('get_skipmatch_phrase_candidates - checking if final candidate is match:', candidate.is_match(skipgram_threshold))
    if candidate.is_match(skipgram_threshold):
        if len(candidates) == 0 or not candidate.same_candidate(candidates[-1]):
            candidate.match_string = candidate.get_match_string(text)
            candidates.append(copy.deepcopy(candidate))
        if candidate.shift_start_skip():
            # candidate string is longer than phrase string check if shifting the start creates
            # a better candidate and if so, add that as well
            candidate.match_string = candidate.get_match_string(text)
            candidates.append(copy.deepcopy(candidate))
    if debug > 1:
        print(f'get_skipmatch_phrase_candidates - returning {len(candidates)} candidates')
    return candidates


def get_skipmatch_candidates(text: Dict[str, any], skip_matches: SkipMatches,
                             skipgram_threshold: float, phrase_model: PhraseModel,
                             max_length_variance: int = 1, ignorecase: bool = False,
                             debug: int = 0) -> List[Candidate]:
    """Find all candidate matches for the phrases in a SkipMatches object.

    :param text: the text object to match with phrases
    :type text: Dict[str, any]
    :param skip_matches: a SkipMatches object with matches between a text and a list of phrases
    :type skip_matches: SkipMatches
    :param skipgram_threshold: a threshold for how many skipgrams should match between a phrase and a candidate
    :type skipgram_threshold: float
    :param phrase_model: a phrase model, either as dictionary or as PhraseModel object
    :type phrase_model: PhraseModel
    :param max_length_variance: the maximum difference in length between candidate and phrase
    :type max_length_variance: int
    :param ignorecase: whether to ignore case when matching skip grams
    :type ignorecase: bool
    :param debug: level to show debug information
    :type debug: int
    :return: a list of candidate matches
    :rtype: List[Candidate]
    """
    phrase_candidates = defaultdict(list)
    candidates: List[Candidate] = []
    for phrase in skip_matches.matches:
        if debug > 0:
            print("get_skipmatch_candidates - phrase:", phrase.phrase_string)
        if get_skipset_overlap(phrase, skip_matches) < skipgram_threshold:
            if debug > 0:
                print('get_skipmatch_candidates - below skipgram_threshold:', get_skipset_overlap(phrase, skip_matches))
            continue
        if phrase.phrase_string in phrase_model.is_variant_of:
            match_phrase = phrase_model.is_variant_of[phrase.phrase_string]
        else:
            match_phrase = phrase.phrase_string
        phrase_candidates[match_phrase] += get_skipmatch_phrase_candidates(text, phrase, skip_matches,
                                                                           skipgram_threshold,
                                                                           max_length_variance=max_length_variance,
                                                                           ignorecase=ignorecase, debug=debug)
    for phrase_string in phrase_candidates:
        if debug > 0:
            print(f"get_skipmatch_candidates - phrase_candidates for phrase {phrase_string}:", len(phrase_candidates[phrase_string]))
        filtered_candidates = filter_overlapping_phrase_candidates(phrase_candidates[phrase_string])
        if debug > 0:
            print(f"get_skipmatch_candidates - filtered_candidates for phrase {phrase_string}:", len(filtered_candidates))
            for candidate in filtered_candidates:
                print('\t', candidate.match_string, candidate.match_start_offset, candidate.match_end_offset)
        candidates += filtered_candidates
    if debug > 0:
        print(f'get_skipmatch_candidates - returning {len(candidates)} candidates')
    return candidates

