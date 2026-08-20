"""PhraseMatch and related classes that represent a fuzzy match between a phrase (or a part of
it) and a span of text, along with their JSON (de)serialization.

Algorithms for building and adjusting the offsets of matches live in
:mod:`fuzzy_search.match.match_offsets`; this module only defines the match data models
themselves.
"""

from __future__ import annotations
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Union

from fuzzy_search._version import __version__
import fuzzy_search.tokenization.string as fuzzy_string
from fuzzy_search.phrase.phrase import Phrase


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


###############
# Match class #
###############

class PhraseMatch:
    """A fuzzy match between a phrase (and a specific spelling variant of it) and a string
    found in a text, with its offsets, label(s) and similarity scores."""

    def __init__(self, match_phrase: Phrase, match_variant: Phrase, match_string: str,
                 match_offset: int, ignorecase: bool = False, text_id: Union[None, str] = None,
                 match_scores: dict = None, match_label: Union[str, List[str]] = None,
                 match_id: str = None, levenshtein_similarity: float = None):
        """Create a PhraseMatch.

        :param match_phrase: a phrase object for which a matching string is found in the text
        :param match_variant: a phrase object for the variant that matches the string in the text
        :param match_string: the matching string found in the text
        :param match_offset: the offset of the matching string in the text
        :param ignorecase: boolean flag whether to ignore case differences
        :param text_id: the identifier of the text in which the match is found
        :param match_scores: the similarity scores of the match
        :param match_label: one or more labels to attach to the match
        :param match_id: an optional identifier to use for the match
        :param levenshtein_similarity: an optional precomputed levenshtein similarity score
        """
        validate_match_props(match_phrase, match_variant, match_string, match_offset)
        self.id = match_id if match_id else str(uuid.uuid4())
        self.phrase = match_phrase
        self.label = match_phrase.label
        if match_label:
            self.label = match_label
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
        """Return a debug representation showing the phrase, variant, string, offset and score."""
        return f'PhraseMatch(' + \
            f'phrase: "{self.phrase.phrase_string}", variant: "{self.variant.phrase_string}", ' + \
            f'string: "{self.string}", offset: {self.offset}, ignorecase: {self.ignorecase}, ' + \
            f'levenshtein_similarity: {self.levenshtein_similarity})'

    @property
    def label_list(self) -> List[str]:
        """Return the match's label(s) as a list, regardless of whether it is stored as a
        single string, a list, or None."""
        if isinstance(self.label, str):
            return [self.label]
        elif isinstance(self.label, list):
            return self.label
        else:
            return []

    def has_label(self, label: str):
        """Check whether this match has the given label.

        :param label: a label string
        :type label: str
        :return: whether the match has this label
        :rtype: bool
        """
        if isinstance(self.label, str):
            return label == self.label
        elif isinstance(self.label, list):
            return label in self.label
        else:
            return label in self.label

    def json(self) -> dict:
        """Return a JSON-serializable dictionary representation of the match."""
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
        """Reconstruct a PhraseMatch from its JSON dictionary representation.

        :param match_json: a JSON dictionary as produced by :meth:`json`
        :return: the reconstructed phrase match
        :rtype: PhraseMatch
        """
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
        match_string = self.string.lower() if self.ignorecase else self.string
        phrase_string = self.variant.phrase_string.lower() if self.ignorecase else self.variant.phrase_string
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
        self.character_overlap = fuzzy_string.score_char_overlap_ratio(phrase_string, match_string)
        return self.character_overlap

    def score_ngram_overlap(self) -> float:
        """Return the ngram overlap between the variant phrase_string and the match_string

        :return: the ngram overlap as proportion of the variant phrase string
        :rtype: float
        """
        match_string = self.string.lower() if self.ignorecase else self.string
        phrase_string = self.variant.phrase_string.lower() if self.ignorecase else self.variant.phrase_string
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
        """Turn match object into a W3C Web Annotation representation.

        :return: a W3C Web Annotation dictionary
        :rtype: Dict[str, any]
        """
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
                "name": f"fuzzy-search v{__version__}"
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
    """A PhraseMatch extended with a window of surrounding text (prefix and suffix context)
    taken from the source document."""

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
        """Return a debug representation showing the phrase, variant, string, offset and context."""
        return f'PhraseMatchInContext(' + \
               f'phrase: "{self.phrase.phrase_string}", variant: "{self.variant.phrase_string}",' + \
               f'string: "{self.string}", offset: {self.offset}), context: "{self.context}"'

    def json(self):
        """Return a JSON-serializable dictionary representation including the context."""
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
        """Turn match object into a W3C Web Annotation representation, including a
        TextQuoteSelector with the prefix/exact/suffix context."""
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
    """Reconstruct a PhraseMatch (or PhraseMatchInContext, if context info is present) from its
    JSON dictionary representation.

    :param match_json: a JSON dictionary representation of a phrase match
    :type match_json: dict
    :return: the reconstructed phrase match
    :rtype: PhraseMatch
    """
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
    """Enumerates how a token match relates a text token to a phrase token: no match, a partial
    match within a phrase token, a full match, or a partial match within a text token."""
    NONE = 0
    PARTIAL_OF_PHRASE_TOKEN = 0.5
    FULL = 1
    PARTIAL_OF_TEXT_TOKEN = 1.5
