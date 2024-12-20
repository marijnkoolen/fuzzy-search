from __future__ import annotations
import uuid
import string
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Dict, Iterable, List, Union

import fuzzy_search
import fuzzy_search.tokenization.string as fuzzy_string
from fuzzy_search.match.candidate_match import Candidate
from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.phrase.phrase_model import PhraseModel
from fuzzy_search.tokenization.token import Token


def filter_matches_by_overlap(filtered_matches: List[PhraseMatch], first_best: bool = False,
                              debug: int = 0) -> List[PhraseMatch]:
    """Filter matches by overlapping match string offsets. When there are multiple phrases matching
    with the same character range in the input text, only pick the matches with the highest
    similarity scores. By default, all matches with the highest similarity score are returned.
    Use 'first_best=True' to return only the first best scoring match.
    """
    if debug > 1:
        print(f"phrase_match.filter_matches_by_overlap - filtered_matches: {len(filtered_matches)}")
    sorted_matches = sorted(filtered_matches, key=lambda x: (x.offset, len(x.string)))
    filtered_matches = []
    if debug > 1:
        print(f"phrase_match.filter_matches_by_overlap - sorted_matches: {len(sorted_matches)}")
    overlapping = defaultdict(list)
    if debug > 1:
        print(f"phrase_match.filter_matches_by_overlap - using first_best: {first_best}")
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
                    print(f"phrase_match.filter_matches_by_overlap - best similarity score: {best_sim}")
                for best_match in sorted_matches:
                    if best_match.levenshtein_similarity < best_sim:
                        break
                    if debug > 1:
                        print(f"phrase_match.filter_matches_by_overlap - best match: "
                              f"({offset_length})\t{best_match.phrase.phrase_string}")
                    filtered_matches.append(best_match)
    return filtered_matches


def candidates_to_matches(candidates: List[Candidate], text: dict, phrase_model: PhraseModel,
                          ignorecase: bool = False) -> List[PhraseMatch]:
    matches: List[PhraseMatch] = []
    for candidate in candidates:
        if candidate.phrase.phrase_string in phrase_model.is_variant_of:
            match_phrase_string = phrase_model.is_variant_of[candidate.phrase.phrase_string]
            match_phrase = phrase_model.phrase_index[match_phrase_string]
        else:
            match_phrase = candidate.phrase
        # print('candidates_to_matches - ignorecase:', ignorecase)
        match = PhraseMatch(match_phrase, candidate.phrase,
                            candidate.match_string, candidate.match_start_offset, text_id=text["id"],
                            ignorecase=ignorecase,
                            # match_label=match_phrase.label
                            )
        match.add_scores(skipgram_overlap=candidate.get_skip_count_overlap())
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
    # print('non_word_prefix:', non_word_prefix)
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
    whitespace_only = True if phrase_string[-1] in punctuation else False
    if debug > 2:
        print('\tadjust_match_end_offset - whitespace_only:', whitespace_only)
    phrase_end = map_string(phrase_string[-3:], punctuation)
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
    return calculate_end_shift(phrase_end, match_end, text_suffix, end_offset)


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


###############
# Match class #
###############

class PhraseMatch:
    """

    Attributes
    """

    def __init__(self, match_phrase: Phrase, match_variant: Phrase, match_string: str,
                 match_offset: int, ignorecase: bool = False, text_id: Union[None, str] = None,
                 match_scores: dict = None, match_label: Union[str, List[str]] = None,
                 match_id: str = None, levenshtein_similarity: float = None):
        """

        :param match_phrase: a phrase object for which a matching string is found in the text
        :param match_variant: a phrase object for the variant that matches the string in the text
        :param match_string: the matching string found in the text
        :param match_offset: the offset of the matching string in the text
        :param ignorecase: boolean flag whether to ignore case differences
        :param text_id: the identifier of the text in which the match is found
        :param match_scores: the similarity scores of the match
        :param match_label: one or more labels to attach to the match
        :param match_id: an optional identifier to use for the match
        """
        # print("Match class match_phrase:", match_phrase)
        validate_match_props(match_phrase, match_variant, match_string, match_offset)
        self.id = match_id if match_id else str(uuid.uuid4())
        self.phrase = match_phrase
        self.label = match_phrase.label
        if match_label:
            self.label = match_label
        # if self.label is None:
        #     print(f'PhraseMatch - self.label is None - match_phrase: {match_phrase}')
        #     print(f'PhraseMatch - self.label is None - match_phrase.label: {match_phrase.label}')
        self.metadata = {}
        self.variant = match_variant
        self.string = match_string
        self.ignorecase = ignorecase
        self.offset = match_offset
        self.end = self.offset + len(self.string)
        self.text_id = text_id
        self.character_overlap: Union[None, float] = None
        self.ngram_overlap: Union[None, float] = None
        self.skipgram_overlap: Union[None, float] = None
        self.levenshtein_similarity: Union[None, float] = levenshtein_similarity
        if match_scores:
            self.character_overlap = match_scores['char_match']
            self.ngram_overlap = match_scores['ngram_match']
            self.levenshtein_similarity = match_scores['levenshtein_similarity']
        self.created = datetime.now()

    def __repr__(self):
        return f'PhraseMatch(' + \
            f'phrase: "{self.phrase.phrase_string}", variant: "{self.variant.phrase_string}", ' + \
            f'string: "{self.string}", offset: {self.offset}, ignorecase: {self.ignorecase}, ' + \
            f'levenshtein_similarity: {self.levenshtein_similarity})'

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
        elif isinstance(self.label, list):
            return label in self.label
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
            "ignorecase": self.ignorecase,
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

    @staticmethod
    def from_json(match_json):
        match_phrase = Phrase(phrase=match_json['phrase'])
        match_variant = Phrase(phrase=match_json['variant'])
        return PhraseMatch(match_phrase=match_phrase, match_variant=match_variant,
                           match_string=match_json['string'], match_offset=match_json['offset'],
                           text_id=match_json['text_id'], match_scores=match_json['match_scores'],
                           match_label=match_json['label'], ignorecase=match_json['ignorecase'])

    def add_scores(self, skipgram_overlap: Union[None, float] = None) -> None:
        """Compute overlap and similarity scores between the match variant and the match string
        and add these to the match object.

        :param skipgram_overlap: the overlap in skipgrams between match string and match variant
        :type skipgram_overlap: Union[float, None]
        :return: None
        :rtype: None
        """
        # print('PhraseMatch - ignorecase:', self.ignorecase)
        match_string = self.string.lower() if self.ignorecase else self.string
        phrase_string = self.variant.phrase_string.lower() if self.ignorecase else self.variant.phrase_string
        # print('match_string:', match_string)
        # print('variant.phrase_string:', self.variant.phrase_string)
        self.character_overlap = fuzzy_string.score_char_overlap_ratio(phrase_string, match_string)
        self.ngram_overlap = fuzzy_string.score_ngram_overlap_ratio(phrase_string, match_string,
                                                                    self.variant.ngram_size)
        self.levenshtein_similarity = fuzzy_string.score_levenshtein_similarity_ratio(phrase_string,
                                                                                      match_string)
        if skipgram_overlap is not None:
            self.skipgram_overlap = skipgram_overlap

    def score_character_overlap(self):
        """Return the character overlap between the variant phrase_string and the match_string

        :return: the character overlap as proportion of the variant phrase string
        :rtype: float
        """
        match_string = self.string.lower() if self.ignorecase else self.string
        phrase_string = self.variant.phrase_string.lower() if self.ignorecase else self.variant.phrase_string
        # print('match_string:', match_string)
        # print('variant.phrase_string:', self.variant.phrase_string)
        self.character_overlap = fuzzy_string.score_char_overlap_ratio(phrase_string, match_string)
        return self.character_overlap

    def score_ngram_overlap(self) -> float:
        """Return the ngram overlap between the variant phrase_string and the match_string

        :return: the ngram overlap as proportion of the variant phrase string
        :rtype: float
        """
        match_string = self.string.lower() if self.ignorecase else self.string
        phrase_string = self.variant.phrase_string.lower() if self.ignorecase else self.variant.phrase_string
        # print('match_string:', match_string)
        # print('variant.phrase_string:', self.variant.phrase_string)
        self.ngram_overlap = fuzzy_string.score_ngram_overlap_ratio(phrase_string,
                                                                    match_string, self.phrase.ngram_size)
        return self.ngram_overlap

    def score_levenshtein_similarity(self):
        """Return the levenshtein similarity between the variant phrase_string and the match_string

        :return: the levenshtein similarity as proportion of the variant phrase string
        :rtype: float
        """
        match_string = self.string.lower() if self.ignorecase else self.string
        phrase_string = self.variant.phrase_string.lower() if self.ignorecase else self.variant.phrase_string
        # print('match_string:', match_string)
        # print('variant.phrase_string:', self.variant.phrase_string)
        self.levenshtein_similarity = fuzzy_string.score_levenshtein_similarity_ratio(phrase_string,
                                                                                      match_string)
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
                "name": f"fuzzy-search v{fuzzy_search.__version__}"
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


class MatchType(Enum):
    NONE = 0
    PARTIAL_OF_PHRASE_TOKEN = 0.5
    FULL = 1
    PARTIAL_OF_TEXT_TOKEN = 1.5


class TokenMatch:

    def __init__(self, text_tokens: Union[Token, List[Token]],
                 phrase_tokens: Union[str, List[str]],
                 match_type: MatchType):
        if isinstance(text_tokens, Token):
            text_tokens = (text_tokens, )
        elif isinstance(text_tokens, list):
            text_tokens = tuple(text_tokens)
        if isinstance(phrase_tokens, str):
            phrase_tokens = (phrase_tokens, )
        elif isinstance(phrase_tokens, list):
            phrase_tokens = tuple(phrase_tokens)
        self.text_tokens = text_tokens
        self.phrase_tokens = phrase_tokens
        self.match_type = match_type
        self.first = text_tokens[0] if isinstance(text_tokens, Iterable) else text_tokens
        self.last = text_tokens[-1] if isinstance(text_tokens, Iterable) else text_tokens
        self.text_start = self.first.char_index
        self.text_end = self.last.char_index + len(self.last)
        self.text_length = self.text_end - self.text_start

    def __repr__(self):
        return f"{self.__class__.__name__}(match_type={self.match_type}, " \
               f"text_tokens={self.text_tokens}, phrase_tokens={self.phrase_tokens})"


class PartialPhraseMatch:

    def __init__(self, phrase: Phrase, token_matches: List[TokenMatch] = None, max_char_gap: int = 20,
                 max_token_gap: int = 1):
        # create a new list instead of pointing to original list
        self.token_matches = []
        self.phrase = phrase
        self.text_tokens = []
        self.phrase_tokens = []
        self.text_phrase_map = defaultdict(list)
        self.missing_tokens = [token.n for token in phrase.tokens]
        self.redundant_tokens = []
        self.max_char_gap = max_char_gap
        self.max_token_gap = max_token_gap
        self.text_start = -1
        self.text_end = -1
        self.text_length = 0
        self.match_string = None
        self.first_text_token = None
        self.last_text_token = None
        self.first_phrase_token = None
        self.last_phrase_token = None
        self.levenshtein_score = None
        if token_matches is not None:
            self.add_tokens(token_matches)

    def __repr__(self):
        return f"{self.__class__.__name__}(\n\tphrase={self.phrase}, \n\ttoken_matches={self.token_matches}, " \
               f"\n\ttext_tokens={self.text_tokens}, \n\tphrase_tokens={self.phrase_tokens}, " \
               f"\n\tmissing_tokens={self.missing_tokens}\n)"

    def _update(self):
        text_tokens = []
        prev_match = None
        for match in self.token_matches:
            if prev_match is None:
                text_tokens.extend(match.text_tokens)
            elif match.text_start == prev_match.text_start:
                continue
            elif match.text_start >= prev_match.text_end:
                text_tokens.extend(match.text_tokens)
            else:
                print('TO DO: figure out how to filter text tokens in partially overlapping token matches')
            # print('_update - match.text_start', match.text_start)
            prev_match = match
        # print('text_tokens:', text_tokens)
        self.text_tokens = tuple(text_tokens)
        # self.text_tokens = tuple([token for match in self.token_matches for token in match.text_tokens])
        self.phrase_tokens = tuple([token for match in self.token_matches for token in match.phrase_tokens])
        self.first_text_token = self.text_tokens[0]
        self.last_text_token = self.text_tokens[-1]
        self.text_start = self.first_text_token.char_index
        self.text_end = self.last_text_token.char_index + len(self.last_text_token)
        self.text_length = self.text_end - self.text_start

    def pop(self):
        self.token_matches.pop(0)
        self._update()

    def _check_gap(self, token_match: TokenMatch):
        token_gap = token_match.text_tokens[0].index - self.text_tokens[-1].index
        char_gap = token_match.text_tokens[0].char_index - self.text_end
        if token_gap > self.max_token_gap or char_gap > self.max_char_gap:
            self.__init__(phrase=self.phrase)

    def push(self, token_match: TokenMatch):
        self.token_matches.append(token_match)
        if len(self.text_tokens) > 0:
            self._check_gap(token_match)
        for text_token in token_match.text_tokens:
            self.text_tokens.append(text_token)
            self.text_phrase_map[text_token].extend(list(token_match.phrase_tokens))
        for phrase_token in token_match.phrase_tokens:
            self.phrase_tokens.append(phrase_token)
            if phrase_token in self.missing_tokens:
                self.missing_tokens.remove(phrase_token)
            else:
                self.redundant_tokens.append(phrase_token)

    def add_tokens(self, token_matches: Union[List[TokenMatch], TokenMatch]):
        if isinstance(token_matches, TokenMatch):
            token_matches = [token_matches]
        for token_match in token_matches:
            for phrase_token in token_match.phrase_tokens:
                if phrase_token in self.missing_tokens:
                    self.missing_tokens.remove(phrase_token)
        self.token_matches.extend(token_matches)
        self._update()


def copy_partial_match(partial_match: PartialPhraseMatch):
    new_pm = PartialPhraseMatch(phrase=partial_match.phrase, token_matches=None,
                                max_char_gap=partial_match.max_char_gap,
                                max_token_gap=partial_match.max_token_gap)
    new_pm.token_matches = [tm for tm in partial_match.token_matches]
    new_pm.missing_tokens = [token for token in partial_match.missing_tokens]
    new_pm.text_tokens = [token for token in partial_match.text_tokens]
    new_pm.phrase_tokens = [token for token in partial_match.phrase_tokens]
    new_pm.redundant_tokens = [token for token in partial_match.redundant_tokens]
    new_pm.first_text_token = partial_match.first_text_token
    new_pm.last_text_token = partial_match.last_text_token
    new_pm.text_start = partial_match.text_start
    new_pm.text_end = partial_match.text_end
    new_pm.text_length = partial_match.text_length
    return new_pm
