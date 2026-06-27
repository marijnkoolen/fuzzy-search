"""Algorithms for building :class:`~fuzzy_search.match.phrase_match.PhraseMatch` objects from
candidate matches, validating their properties, and adjusting their start/end offsets to align
with word boundaries."""

import string
from collections import defaultdict
from typing import Dict, List, Union

import fuzzy_search.tokenization.string as fuzzy_string
from fuzzy_search.match.candidate_match import Candidate
from fuzzy_search.match.phrase_match import PhraseMatch
from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.phrase.phrase_model import PhraseModel


def filter_matches_by_overlap(filtered_matches: List[PhraseMatch], first_best: bool = False,
                              debug: int = 0) -> List[PhraseMatch]:
    """Filter matches by overlapping match string offsets. When there are multiple phrases matching
    with the same character range in the input text, only pick the matches with the highest
    similarity scores. By default, all matches with the highest similarity score are returned.
    Use 'first_best=True' to return only the first best scoring match.
    """
    if debug > 1:
        print(f"match_offsets.filter_matches_by_overlap - filtered_matches: {len(filtered_matches)}")
    sorted_matches = sorted(filtered_matches, key=lambda x: (x.offset, len(x.string)))
    filtered_matches = []
    if debug > 1:
        print(f"match_offsets.filter_matches_by_overlap - sorted_matches: {len(sorted_matches)}")
    overlapping = defaultdict(list)
    if debug > 1:
        print(f"match_offsets.filter_matches_by_overlap - using first_best: {first_best}")
    for match in sorted_matches:
        overlapping[(match.offset, len(match.string))].append(match)
    for offset_length in overlapping:
        if len(overlapping[offset_length]) == 1:
            filtered_matches.extend(overlapping[offset_length])
        else:
            if first_best is True:
                first_best = max(overlapping[offset_length], key=lambda item: item.levenshtein_similarity)
                filtered_matches.append(first_best)
            else:
                sorted_matches = sorted(overlapping[offset_length], key=lambda item: item.levenshtein_similarity,
                                        reverse=True)
                best_sim = sorted_matches[0].levenshtein_similarity
                if debug > 1:
                    print(f"match_offsets.filter_matches_by_overlap - best similarity score: {best_sim}")
                for best_match in sorted_matches:
                    if best_match.levenshtein_similarity < best_sim:
                        break
                    if debug > 1:
                        print(f"match_offsets.filter_matches_by_overlap - best match: "
                              f"({offset_length})\t{best_match.phrase.phrase_string}")
                    filtered_matches.append(best_match)
    return filtered_matches


def candidates_to_matches(candidates: List[Candidate], text: dict, phrase_model: PhraseModel,
                          ignorecase: bool = False) -> List[PhraseMatch]:
    """Convert a list of fuzzy match candidates into PhraseMatch objects, resolving variant
    candidates to their corresponding main phrase and computing similarity scores.

    :param candidates: a list of candidate matches
    :type candidates: List[Candidate]
    :param text: the text object the candidates were found in
    :type text: dict
    :param phrase_model: the phrase model containing the phrases and their variants
    :type phrase_model: PhraseModel
    :param ignorecase: whether to ignore case when scoring matches
    :type ignorecase: bool
    :return: a list of phrase matches
    :rtype: List[PhraseMatch]
    """
    matches: List[PhraseMatch] = []
    for candidate in candidates:
        if candidate.phrase.phrase_string in phrase_model.is_variant_of:
            match_phrase_string = phrase_model.is_variant_of[candidate.phrase.phrase_string]
            match_phrase = phrase_model.phrase_index[match_phrase_string]
        else:
            match_phrase = candidate.phrase
        match = PhraseMatch(match_phrase, candidate.phrase,
                            candidate.match_string, candidate.match_start_offset, text_id=text["id"],
                            ignorecase=ignorecase)
        match.add_scores(skipgram_overlap=candidate.skipgram_overlap)
        matches.append(match)
    return matches


def validate_match_props(match_phrase: Phrase, match_variant: Phrase,
                         match_string: str, match_offset: int) -> None:
    """Validate match properties.

    :param match_phrase: the phrase that has been matched
    :type match_phrase: Phrase
    :param match_variant: the variant of the phrase that the match is based on
    :type match_variant: Phrase
    :param match_string: the text string that matches the variant phrase
    :type match_string: str
    :param match_offset: the offset of the match string in the text
    :type match_offset: int
    :return: None
    :rtype: None
    """
    if not isinstance(match_phrase, Phrase):
        print(f"match_phrase: {match_phrase}")
        print(f"type: {type(match_phrase)}")
        raise TypeError('match_phrase MUST be of class Phrase')
    if not isinstance(match_variant, Phrase):
        raise TypeError('match_variant MUST be of class Phrase')
    if not isinstance(match_string, str):
        raise TypeError('match string MUST be a string')
    if len(match_string) == 0:
        print(f"match_phrase: {match_phrase}")
        raise ValueError('match string cannot be empty string')
    if not isinstance(match_offset, int):
        raise TypeError('match_offset must be an integer')
    if match_offset < 0:
        raise ValueError('offset cannot be negative')


def adjust_match_start_offset(text: Dict[str, any], match_string: str,
                              match_offset: int) -> Union[int, None]:
    """Adjust the start offset if it is not at a word boundary.

    :param text: the text object that contains the candidate match string
    :type text: Dict[str, any]
    :param match_string: the candidate match string
    :type match_string: str
    :param match_offset: the text offset of the candidate match string
    :type match_offset: int
    :return: the adjusted offset or None if the required adjustment is too big
    :rtype: Union[int, None]
    """
    # adjust the start
    # check if there match initial is a non-word character
    non_word_prefix = fuzzy_string.get_non_word_prefix(match_string)
    if non_word_prefix == "":
        # match does not start with a non-word prefix, so check if it needs to be moved to the left
        if match_offset == 0:
            # match is at the start of text and starts with word characters
            return match_offset
        # if character before match is first of text and not a word boundary, move left
        if match_offset == 1 and text['text'][0] not in fuzzy_string.non_word_affixes_1:
            return match_offset - 1
        # if character before match is a word boundary, match offset is good
        if text["text"][match_offset-1:match_offset] in fuzzy_string.non_word_affixes_1:
            return match_offset
        # if penultimate character before match is a word boundary, move offset by -1
        elif match_offset > 1 and text["text"][match_offset-2:match_offset-1] in fuzzy_string.non_word_affixes_1:
            # move match_offset back by 1 to start at word boundary
            return match_offset-1
        # if penultimate character before match is start of text, move offset by -2
        elif match_offset == 2 and text['text'][0] not in fuzzy_string.non_word_affixes_1:
            return match_offset - 2
        # if two characters before match is a word boundary, move offset by -2
        elif match_offset > 2 and text["text"][match_offset-3:match_offset-2] in fuzzy_string.non_word_affixes_1:
            # move match_offset back by 1 to start at word boundary
            return match_offset-2
        # if the three characters preceding match are word characters, the start is wrong
        else:
            return None
    else:
        # match starts with a non-word-prefix, so move offset to after the prefix
        return match_offset + len(non_word_prefix)


def adjust_match_end_offset(phrase_string: str, candidate_string: str,
                            text: Dict[str, any], end_offset: int, punctuation: str,
                            debug: int = 0) -> Union[int, None]:
    """Adjust the end offset if it is not at a word boundary.

    :param phrase_string: the phrase string
    :type phrase_string: str
    :param candidate_string: the candidate match string
    :type candidate_string: str
    :param text: the text object that contains the candidate match string
    :type text: Dict[str, any]
    :param end_offset: the text offset of the candidate match string
    :type end_offset: int
    :param punctuation: the set of characters to treat as punctuation
    :type punctuation: str
    :param debug: level to show debug information
    :type debug: int
    :return: the adjusted offset or None if the required adjustment is too big
    :rtype: Union[int, None]
    """
    # ugly hack: if phrase string ends with punctuation, use only whitespace as word end boundary
    if debug > 2:
        print('\tadjust_match_end_offset - start')
    if phrase_string[-1] in punctuation:
        whitespace_only = True
    elif phrase_string[-1] in ' \t\r\n' and phrase_string[-2] in punctuation:
        whitespace_only = True
    else:
        whitespace_only = False
    if debug > 2:
        print('\tadjust_match_end_offset - whitespace_only:', whitespace_only)
    phrase_end = map_string(phrase_string[-3:], punctuation, whitespace_only=whitespace_only)
    if debug > 2:
        print('\tadjust_match_end_offset - prhase_end:', phrase_end)
    match_end = map_string(candidate_string[-3:], punctuation, whitespace_only=whitespace_only)
    if debug > 2:
        print('\tadjust_match_end_offset - match_end:', match_end)
    text_suffix = map_string(text["text"][end_offset:end_offset+3], punctuation,
                             whitespace_only=whitespace_only, debug=debug)
    if debug > 2:
        print('\tadjust_match_end_offset - text_suffix:', text_suffix)
        print(f"\tadjust_match_end_offset - match_end: {candidate_string[-3:]: <4}\ttext_suffix: {text['text'][end_offset:end_offset+3]: >4}")
        print(f"\tadjust_match_end_offset - mapped suffixes - match_end: #{match_end}#\ttext_suffix: #{text_suffix}#")
    try:
        return calculate_end_shift(phrase_end, match_end, text_suffix, end_offset)
    except ValueError:
        print(f"phrase_string: #{phrase_string}#\tcandidate_string: #{candidate_string}#")
        print(f"text: #{text}#")
        print(f"text_suffix: #{text_suffix}#")
        print(f"phrase_end: #{phrase_end}#")
        print(f"match_end: #{match_end}#")
        print(f"whitespace_only: #{whitespace_only}#")
        raise


def adjust_match_offsets(phrase_string: str, candidate_string: str,
                         text: Dict[str, any], candidate_start_offset: int,
                         candidate_end_offset: int,
                         punctuation: str = string.punctuation,
                         debug: int = 0) -> Union[Dict[str, Union[str, int]], None]:
    """Adjust the end offset if it is not at a word boundary.

    :param phrase_string: the phrase string
    :type phrase_string: str
    :param candidate_string: the candidate match string
    :type candidate_string: str
    :param text: the text object that contains the candidate match string
    :type text: Dict[str, any]
    :param candidate_start_offset: the text offset of the start of the candidate match string
    :type candidate_start_offset: int
    :param candidate_end_offset: the text offset of the end of the candidate match string
    :type candidate_end_offset: int
    :param punctuation: the set of characters to treat as punctuation (defaults to string.punctuation)
    :type punctuation: str
    :param debug: level to show debug information
    :type debug: int
    :return: the adjusted offset or None if the required adjustment is too big
    :rtype: Union[int, None]
    """
    if debug > 2:
        print("\tadjust_match_offset - phrase string:", phrase_string)
        print("\tadjust_match_offset - adjusting candidate string:", candidate_string)
    if punctuation is None:
        punctuation = string.punctuation
    if debug > 2:
        print("\tadjust_match_offset - candidate_start_offset:", candidate_start_offset)
    match_start_offset = adjust_match_start_offset(text, candidate_string, candidate_start_offset)
    if debug > 2:
        print("\tadjust_match_offset - match_start_offset:", match_start_offset)
    if match_start_offset is None:
        return None
    match_end_offset = adjust_match_end_offset(phrase_string, candidate_string,
                                               text, candidate_end_offset, punctuation, debug=debug)
    if debug > 2:
        print("\tadjust_match_offset - match_end_offset:", match_end_offset)
    if match_end_offset is None:
        return None
    elif match_end_offset <= match_start_offset:
        return None
    return {
        "match_string": text["text"][match_start_offset:match_end_offset],
        "match_start_offset": match_start_offset,
        "match_end_offset": match_end_offset
    }


def map_string(affix_string: str, punctuation: str,
               whitespace_only: bool = False, debug: int = 0) -> str:
    """Turn affix string into type char representation. Types are 'w' for non-whitespace char,
    and 's' for whitespace char.

    :param affix_string: a string
    :type: str
    :param punctuation: the set of characters to treat as punctuation
    :type punctuation: str
    :param whitespace_only: whether to treat only whitespace as word boundary or also include (some) punctuation
    :type whitespace_only: bool
    :param debug: level to show debug information
    :type debug: int
    :return: the type char representation
    :rtype: str
    """
    if whitespace_only:
        return ''.join(['s' if char in ' \t\n\r' else 'w' for char in affix_string])
    else:
        return ''.join(['s' if char in ' \t\n\r' or char in punctuation else 'w' for char in affix_string])


def calculate_end_shift(phrase_end: str, match_end: str, text_suffix: str, end_offset: int):
    """Determine whether and how much to shift the end offset, based on trailing whitespace
    for either the phrase or the match or both."""
    if phrase_end == match_end:
        if text_suffix == "" or text_suffix.startswith("s"):
            return end_offset
    if phrase_end.endswith("s") and match_end.endswith("s"):
        # both phrase and match end in whitespace, so no need to shift
        return end_offset
    if match_end == "wss":
        return end_offset - 2
    if phrase_end == "www":
        if match_end == "www":
            if text_suffix == "w" or text_suffix.startswith("ws"):
                return end_offset + 1
            elif text_suffix == "ww" or text_suffix.startswith("wws"):
                return end_offset + 2
            elif text_suffix.startswith("www"):
                return None
        if match_end == "wws":
            return end_offset - 1
        if match_end == "wsw":
            if text_suffix == "" or text_suffix.startswith("s"):
                # we assume the whitespace in the match is a misrecognised word character
                return end_offset
            if text_suffix.startswith("w"):
                # we assume the whitespace in the match is correct
                return end_offset - 2
        if match_end == "sww":
            if text_suffix == "" or text_suffix.startswith("s"):
                # we assume the whitespace in the match is a misrecognised word character
                return end_offset
            elif text_suffix.startswith("w"):
                # we assume the whitespace in the match is correct
                return None
        if match_end == "sws":
            # we assume the first whitespace in the match is a misrecognised word character
            return end_offset - 1
        if match_end == "ssw":
            return None
        else:
            return None
    if phrase_end == "wws":
        if match_end == "www":
            if text_suffix == "":
                return end_offset
            elif text_suffix.startswith("s"):
                return end_offset + 1
            elif text_suffix == "w":
                return None
            elif text_suffix.startswith("ws"):
                return end_offset + 2
            elif text_suffix.startswith("ww"):
                return None
            else:
                return None
        elif match_end.startswith("ws"):
            return end_offset - 1
        elif match_end.startswith("s"):
            return end_offset - 2
        else:
            return None
    if phrase_end == "sww":
        if match_end == "sww":
            if text_suffix == "w" or text_suffix.startswith("ws"):
                return end_offset + 1
            else:
                return None
        elif match_end == "sws":
            return end_offset - 1
        elif match_end == "www":
            if text_suffix == "" or text_suffix.startswith("s"):
                return end_offset
            else:
                return None
        elif match_end == "wsw":
            if text_suffix == "" or text_suffix.startswith("s"):
                return end_offset
            if text_suffix == "ws" or text_suffix.startswith("ws"):
                return end_offset + 1
            if text_suffix == "ww" or text_suffix.startswith("wws"):
                return end_offset + 2
            else:
                return None
        elif match_end == "ssw":
            if text_suffix == "" or text_suffix.startswith("s"):
                return end_offset
            elif text_suffix == "w" or text_suffix.startswith("ws"):
                return end_offset + 1
            elif text_suffix == "ww" or text_suffix.startswith("wws"):
                return end_offset + 1
            else:
                return None
        else:
            return None
    if phrase_end == "sws":
        if match_end == "www":
            if text_suffix == "sw" or text_suffix == "sws":
                return end_offset + 2
            else:
                return None
        elif match_end == "sww":
            return end_offset - 2
        elif match_end == "wsw":
            if text_suffix == "":
                return end_offset
            if text_suffix.startswith("s"):
                return end_offset + 1
            else:
                return end_offset - 1
        else:
            return None
    if phrase_end == "wsw":
        if match_end == "wsw":
            if text_suffix == "w" or text_suffix.startswith("ws"):
                return end_offset + 1
            else:
                return None
        if match_end == "www":
            if text_suffix == "" or text_suffix == "s":
                return end_offset
            elif text_suffix.startswith("w"):
                return None
            elif text_suffix == "sw" or text_suffix == "sws":
                return end_offset + 2
            else:
                return None
        if match_end == "sww":
            if text_suffix == "":
                return end_offset + 1
            elif text_suffix.startswith("s"):
                return end_offset
            else:
                return None
        if match_end == "ssw":
            if text_suffix == "" or text_suffix.startswith("s"):
                return end_offset
            else:
                return None
        else:
            return None
    if len(phrase_end) < 3:
        if phrase_end == match_end:
            return end_offset
        else:
            return None
    else:
        details = f"phrase_end {phrase_end}, match_end {match_end}, text_suffix {text_suffix}"
        raise ValueError(f"combination not captured: {details}")
