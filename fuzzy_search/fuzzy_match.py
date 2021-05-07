from __future__ import annotations
from typing import Dict, List, Union
from datetime import datetime
from collections import Counter
import uuid
import string

import fuzzy_search.fuzzy_string as fuzzy_string
from fuzzy_search.fuzzy_string import SkipGram
from fuzzy_search.fuzzy_phrase import Phrase


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
        print(match_phrase)
        print(type(match_phrase))
        raise TypeError('match_phrase MUST be of class Phrase')
    if not isinstance(match_variant, Phrase):
        raise TypeError('match_variant MUST be of class Phrase')
    if not isinstance(match_string, str):
        raise TypeError('match string MUST be a string')
    if len(match_string) == 0:
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
        # if character before match is a word boundary, match offset is good
        if text["text"][match_offset-1:match_offset] in fuzzy_string.non_word_affixes_1:
            return match_offset
        # if penultimate character before match is a word boundary, move offset by -1
        elif match_offset > 1 and text["text"][match_offset-2:match_offset-1] in fuzzy_string.non_word_affixes_1:
            # move match_offset back by 1 to start at word boundary
            return match_offset-1
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
                            text: Dict[str, any], end_offset: int, punctuation: str) -> Union[int, None]:
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
    :return: the adjusted offset or None if the required adjustment is too big
    :rtype: Union[int, None]
    """
    # ugly hack: if phrase string ends with punctuation, use only whitespace as word end boundary
    whitespace_only = True if phrase_string[-1] in punctuation else False
    phrase_end = map_string(phrase_string[-3:], punctuation)
    match_end = map_string(candidate_string[-3:], punctuation, whitespace_only=whitespace_only)
    text_suffix = map_string(text["text"][end_offset:end_offset+3], punctuation, whitespace_only=whitespace_only)
    # print(f"match_end: {candidate_string[-3:]: <4}\ttext_suffix: {text['text'][end_offset:end_offset+3]: >4}")
    # print(f"mapped suffixes - match_end: #{match_end}#\ttext_suffix: #{text_suffix}#")
    return calculate_end_shift(phrase_end, match_end, text_suffix, end_offset)


def adjust_match_offsets(phrase_string: str, candidate_string: str,
                         text: Dict[str, any], candidate_start_offset: int,
                         candidate_end_offset: int,
                         punctuation: str = string.punctuation) -> Union[Dict[str, Union[str, int]], None]:
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
    :return: the adjusted offset or None if the required adjustment is too big
    :rtype: Union[int, None]
    """
    # print("phrase string:", phrase_string)
    # print("adjusting candidate string:", candidate_string)
    if punctuation is None:
        punctuation = string.punctuation
    # print("candidate_start_offset:", candidate_start_offset)
    match_start_offset = adjust_match_start_offset(text, candidate_string, candidate_start_offset)
    # print("match_start_offset:", match_start_offset)
    if match_start_offset is None:
        return None
    match_end_offset = adjust_match_end_offset(phrase_string, candidate_string,
                                               text, candidate_end_offset, punctuation)
    if match_end_offset is None:
        return None
    return {
        "match_string": text["text"][match_start_offset:match_end_offset],
        "match_start_offset": match_start_offset,
        "match_end_offset": match_end_offset
    }


def map_string(affix_string: str, punctuation: str, whitespace_only: bool = False) -> str:
    """Turn affix string into type char representation. Types are 'w' for non-whitespace char,
    and 's' for whitespace char.

    :param affix_string: a string
    :type: str
    :param punctuation: the set of characters to treat as punctuation
    :type punctuation: str
    :param whitespace_only: whether to treat only whitespace as word boundary or also include (some) punctuation
    :type whitespace_only: bool
    :return: the type char representation
    :rtype: str
    """
    if whitespace_only:
        return "".join(["s" if char == " " else "w" for char in affix_string])
    else:
        return "".join(["s" if char == " " or char in punctuation else "w" for char in affix_string])


def calculate_end_shift(phrase_end: str, match_end: str, text_suffix: str, end_offset: int):
    if phrase_end == match_end:
        if text_suffix == "" or text_suffix.startswith("s"):
            return end_offset
    if phrase_end.endswith("s") and match_end.endswith("s"):
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


###################
# Candidate class #
###################

class Candidate:

    def __init__(self, phrase: Phrase, max_length_variance: int = 1):
        """Create a Candidate instance for a given Phrase object.

        :param phrase: a phrase object
        :type phrase: Phrase
        """
        self.skipgram_set = set()
        self.skipgram_list: List[SkipGram] = []
        self.skipgram_count = Counter()
        self.phrase = phrase
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
        if len(self.skipgram_list) == 0 and skipgram.string not in self.phrase.early_skipgram_index:
            # print("skipping skipgram as first for candidate:", skipgram.string)
            return None
        self.skipgram_set.add(skipgram.string)
        self.skipgram_list.append(skipgram)
        if self.match_start_offset is None or self.match_start_offset < 0:
            self.match_start_offset = self.get_match_start_offset()
        if skipgram.offset + skipgram.length > self.match_end_offset:
            self.match_end_offset = skipgram.offset + skipgram.length
        self.skipgram_count.update([skipgram.string])
        # print("\tadd - skipgram:", skipgram.string, skipgram.offset)
        # print("\tadd - match length:", self.skip_match_length())
        # print("\tadd - list:", [skip.string for skip in self.skipgram_list])
        # check if the candidate string is too long to match the phrase
        # if too long, remove the first skipgrams until the string is short enough
        while self.skip_match_length() > self.max_length and len(self.skipgram_list) > 0:
            self.remove_first_skip()
            self.match_start_offset = self.get_match_start_offset()
            # print("\tremove - too long - length:", self.skip_match_length())
            # print("\tremove - too long - list:", [skip.string for skip in self.skipgram_list])
            # print("\tremove - too long - start:", self.match_start_offset, "\tend:", self.match_end_offset)
        while len(self.skipgram_list) > 0 and self.skipgram_list[0].string not in self.phrase.early_skipgram_index:
            self.remove_first_skip()
            self.match_start_offset = self.get_match_start_offset()
            # print("\tremove - no start - length:", self.skip_match_length())
            # print("\tremove - no start - list:", [skip.string for skip in self.skipgram_list])
            # print("\tremove - no start - start:", self.match_start_offset, "\tend:", self.match_end_offset)

    def shift_start_skip(self) -> bool:
        """Check if there is a later skip that is a better start."""
        if self.skip_match_length() <= len(self.phrase.phrase_string):
            return False
        start_skip = self.skipgram_list[0]
        start_phrase_offset = self.phrase.skipgram_index[start_skip.string][0].offset
        best_start_phrase_offset = start_phrase_offset
        best_start_index = 0
        best_start_skip = start_skip
        for si, skip in enumerate(self.skipgram_list):
            skip_phrase_offset = self.phrase.skipgram_index[skip.string][0].offset
            if skip.offset - start_skip.offset > self.skip_match_length() - len(self.phrase.phrase_string):
                # stop looking for better start when remaining skips result in too short match length
                break
            if skip.offset > best_start_skip.offset and skip_phrase_offset <= best_start_phrase_offset:
                best_start_index = si
                best_start_skip = skip
                best_start_phrase_offset = skip_phrase_offset
            if skip.string not in self.phrase.early_skipgram_index:
                break
        for _ in range(0, best_start_index):
            self.remove_first_skip()
        self.match_start_offset = self.get_match_start_offset()
        return best_start_index > 0

    def remove_first_skip(self) -> None:
        """Remove the first matching skipgram from the list and update the count and set."""
        first_skip = self.skipgram_list.pop(0)
        # print('removing first skip')
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
            # print('NO MATCH: there are no matching skipgrams, so no matching string')
            return False
        if self.skipgram_list[0].string not in self.phrase.early_skipgram_index:
            # the first skipgram of candidate is not in the early skipgrams of phrase
            # print('NO MATCH: the first skipgram of candidate is not in the early skipgrams of phrase')
            return False
        # length of current skip matches should be no longer than phrase plus max length variance
        # but also no shorter than phrase minus max length variance minus late skips offset from end
        # of phrase.
        if self.skip_match_length() > len(self.phrase.phrase_string) + self.max_length_variance:
            # print('NO MATCH: skip match length is longer than max length variance of phrase')
            return False
        elif self.skip_match_length() < self.phrase.late_threshold - self.max_length_variance:
            # print('phrase length:', len(self.phrase.phrase_string))
            # print('skip match length:', self.skip_match_length())
            # print('max length variance:', self.max_length_variance)
            # print('late threshold:', self.phrase.late_threshold)
            # print('NO MATCH: skip match length is not with max length variance of phrase')
            return False
        # print(self.get_skip_set_overlap())
        # print('late threshold:', self.phrase.late_threshold)
        if self.skipgram_list[-1].string not in self.phrase.late_skipgram_index:
            # print("NO MATCH: last skip not in late index")
            # the last skipgram of candidate is not in the late skipgrams of phrase
            return False
        if self.get_skip_set_overlap() < skipgram_threshold:
            # print("below skipgram threshold:", self.get_skip_set_overlap(), skipgram_threshold)
            return False
        else:
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
            diff += abs(count - self.phrase.skipgram_freq[skipgram_string])
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
        first_skip_in_phrase = self.phrase.skipgram_index[first_skip.string][0]
        match_start_offset = self.skipgram_list[0].offset - first_skip_in_phrase.offset
        # print("in match:", first_skip.string, first_skip.offset)
        # print("in phrase:", first_skip_in_phrase.string, first_skip_in_phrase.offset)
        # print("match_start_offset:", match_start_offset)
        return 0 if match_start_offset < 0 else match_start_offset

    def get_match_string(self, text: Dict[str, any]) -> Union[str, None]:
        """Find the matching string of a candidate fuzzy match between a text and a phrase.

        :param text: the text object from which the candidate was derived
        :type text: Dict[str, any]
        :return: the matching string
        :rtype: str
        """
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


###############
# Match class #
###############

class PhraseMatch:

    def __init__(self, match_phrase: Phrase, match_variant: Phrase, match_string: str,
                 match_offset: int, text_id: Union[None, str] = None,
                 match_scores: dict = None, match_label: Union[str, List[str]] = None,
                 match_id: str = None):
        # print("Match class match_phrase:", match_phrase)
        validate_match_props(match_phrase, match_variant, match_string, match_offset)
        self.id = match_id if match_id else str(uuid.uuid4())
        self.phrase = match_phrase
        self.label = match_phrase.label
        if match_label:
            self.label = match_label
        self.metadata = {}
        self.variant = match_variant
        self.string = match_string
        self.offset = match_offset
        self.end = self.offset + len(self.string)
        self.text_id = text_id
        self.character_overlap: Union[None, float] = None
        self.ngram_overlap: Union[None, float] = None
        self.skipgram_overlap: Union[None, float] = None
        self.levenshtein_similarity: Union[None, float] = None
        if match_scores:
            self.character_overlap = match_scores['char_match']
            self.ngram_overlap = match_scores['ngram_match']
            self.levenshtein_similarity = match_scores['levenshtein_similarity']
        self.created = datetime.now()

    def __repr__(self):
        return f'PhraseMatch(' + \
            f'phrase: "{self.phrase.phrase_string}", variant: "{self.variant.phrase_string}",' + \
            f'string: "{self.string}", offset: {self.offset})'

    @property
    def label_list(self) -> List[str]:
        if isinstance(self.label, str):
            return [self.label]
        elif isinstance(self.label, list):
            return self.label
        else:
            return []

    def has_label(self, label: str):
        if isinstance(self.label, str):
            return label == self.label
        else:
            return label in self.label

    def json(self) -> dict:
        data = {
            "type": "PhraseMatch",
            "phrase": self.phrase.phrase_string,
            "variant": self.variant.phrase_string,
            "string": self.string,
            "offset": self.offset,
            "label": self.label,
            "text_id": self.text_id,
            "match_scores": {
                "char_match": self.character_overlap,
                "ngram_match": self.ngram_overlap,
                "levenshtein_similarity": self.levenshtein_similarity
            }
        }
        if "label" in self.phrase.metadata:
            data["label"] = self.phrase.metadata["label"]
        return data

    def add_scores(self, skipgram_overlap: Union[None, float] = None) -> None:
        """Compute overlap and similarity scores between the match variant and the match string
        and add these to the match object.

        :param skipgram_overlap: the overlap in skipgrams between match string and match variant
        :type skipgram_overlap: Union[float, None]
        :return: None
        :rtype: None
        """
        self.character_overlap = self.score_character_overlap()
        self.ngram_overlap = self.score_ngram_overlap()
        self.levenshtein_similarity = self.score_levenshtein_similarity()
        if skipgram_overlap is not None:
            self.skipgram_overlap = skipgram_overlap

    def score_character_overlap(self):
        """Return the character overlap between the variant phrase_string and the match_string

        :return: the character overlap as proportion of the variant phrase string
        :rtype: float
        """
        if not self.character_overlap:
            self.character_overlap = fuzzy_string.score_char_overlap_ratio(self.variant.phrase_string, self.string)
        return self.character_overlap

    def score_ngram_overlap(self) -> float:
        """Return the ngram overlap between the variant phrase_string and the match_string

        :return: the ngram overlap as proportion of the variant phrase string
        :rtype: float
        """
        if not self.ngram_overlap:
            self.ngram_overlap = fuzzy_string.score_ngram_overlap_ratio(self.variant.phrase_string,
                                                                        self.string, self.phrase.ngram_size)
        return self.ngram_overlap

    def score_levenshtein_similarity(self):
        """Return the levenshtein similarity between the variant phrase_string and the match_string

        :return: the levenshtein similarity as proportion of the variant phrase string
        :rtype: float
        """
        if not self.levenshtein_similarity:
            self.levenshtein_similarity = fuzzy_string.score_levenshtein_similarity_ratio(self.variant.phrase_string,
                                                                                          self.string)
        return self.levenshtein_similarity

    def overlaps(self, other: PhraseMatch) -> bool:
        """Check if the match string of this match object overlaps with the match string of another match object.

        :param other: another match object
        :type other: PhraseMatch
        :return: a boolean indicating whether the match_strings of the two objects overlap in the source text
        :rtype: bool"""
        if self.text_id is not None and self.text_id != other.text_id:
            return False
        if self.offset <= other.offset < self.end:
            return True
        elif other.offset <= self.offset < other.end:
            return True
        else:
            return False

    def as_web_anno(self) -> Dict[str, any]:
        """Turn match object into a W3C Web Annotation representation"""
        if not self.text_id:
            raise ValueError('Cannot make target: match object has no text_id')
        body_match = [
            {
                'type': 'TextualBody',
                'purpose': 'tagging',
                'format': 'text',
                'value': self.phrase.phrase_string
            },
            {
                'type': 'TextualBody',
                'purpose': 'highlighting',
                'format': 'text',
                'value': self.string
            }
        ]
        if self.variant.phrase_string != self.string:
            correction = {
                'type': 'TextualBody',
                'purpose': 'correcting',
                'format': 'text',
                'value': self.variant.phrase_string
            }
            body_match.append(correction)
        if self.label:
            classification = {
                'type': 'TextualBody',
                'purpose': 'classifying',
                'format': 'text',
                'value': self.label
            }
            body_match.append(classification)
        return {
            "@context": "http://www.w3.org/ns/anno.jsonld",
            "id": self.id,
            "type": "Annotation",
            "motivation": "classifying",
            "created": self.created.isoformat(),
            "generator": {
                "id": "https://github.com/marijnkoolen/fuzzy-search",
                "type": "Software",
                "name": f"fuzzy-search"
            },
            "target": {
                "source": self.text_id,
                "selector": {
                    "type": "TextPositionSelector",
                    "start": self.offset,
                    "end": self.end
                }
            },
            "body": body_match
        }


class PhraseMatchInContext(PhraseMatch):

    def __init__(self, match: PhraseMatch, text: Union[str, dict] = None, context: str = None,
                 context_start: int = None, context_end: int = None,
                 prefix_size: int = 20, suffix_size: int = 20):
        super().__init__(match_phrase=match.phrase, match_variant=match.variant, match_string=match.string,
                         match_offset=match.offset, text_id=match.text_id)
        """MatchInContext extends a Match object with surrounding context from the text document that the match
        phrase was taken from. Alternatively, the context can be submitted.

        :param text: the text (string or dictionary with 'text' and 'id' properties) that the match phrase was taken from
        :type text: Union[str, dict]
        :param context: the context string around the match phrase 
        :type context: Union[str, dict]
        :param match: the match phrase object
        :type match: Match
        :param context_start: the start offset of the context in the original text
        :type context_start: int
        :param context_end: the end offset of the context in the original text
        :type context_end: int
        :param prefix_size: the size of the prefix window
        :type prefix_size: int
        :param suffix_size: the size of the suffix window
        :type suffix_size: int 
        """
        self.character_overlap = match.character_overlap
        self.ngram_overlap = match.ngram_overlap
        self.levenshtein_similarity = match.levenshtein_similarity
        self.prefix_size = prefix_size
        self.suffix_size = suffix_size
        if text:
            if isinstance(text, str):
                text = {"text": text, "id": match.text_id}
            self.context_start = match.offset - prefix_size if match.offset >= prefix_size else 0
            self.context_end = match.end + suffix_size if len(text["text"]) > match.end + suffix_size else len(text["text"])
            self.context = text["text"][self.context_start:self.context_end]
        elif context:
            self.context = context
            self.context_start = context_start
            self.context_end = context_end
        self.prefix = text["text"][self.context_start:match.offset]
        self.suffix = text["text"][match.end:self.context_end]

    def __repr__(self):
        return f'PhraseMatchInContext(' + \
               f'phrase: "{self.phrase.phrase_string}", variant: "{self.variant.phrase_string}",' + \
               f'string: "{self.string}", offset: {self.offset}), context: "{self.context}"'

    def json(self):
        json_data = super().json()
        json_data["context_start"] = self.context_start
        json_data["context_end"] = self.context_end
        json_data["context"] = self.context
        json_data['prefix_size'] = self.prefix_size
        json_data['suffix_size'] = self.suffix_size
        json_data['prefix'] = self.prefix
        json_data['suffix'] = self.suffix
        return json_data

    def as_web_anno(self) -> Dict[str, any]:
        match_anno = super().as_web_anno()
        position_selector = match_anno['target']['selector']
        quote_selector = {
            'type': 'TextQuoteSelector',
            'prefix': self.prefix,
            'exact': self.string,
            'suffix': self.suffix
        }
        match_anno['target']['selector'] = [position_selector, quote_selector]
        return match_anno


def phrase_match_from_json(match_json: dict) -> PhraseMatch:
    match_phrase = Phrase(match_json['phrase'])
    match_variant = Phrase(match_json['variant'])
    phrase_match = PhraseMatch(match_phrase, match_variant, match_json['string'],
                               match_offset=match_json['offset'],
                               match_scores=match_json['match_scores'],
                               match_label=match_json['label'])
    if 'context' in match_json:
        phrase_match = PhraseMatchInContext(phrase_match, context=match_json['context'],
                                            prefix_size=match_json['prefix_size'],
                                            suffix_size=match_json['suffix_size'],
                                            context_start=match_json['context_start'],
                                            context_end=match_json['context_end'])
    return phrase_match
