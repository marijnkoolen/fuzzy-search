import copy
import json
import re
from collections import defaultdict
from typing import Dict, Generator, List, Set, Union

from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.tokenization.token import Token
from fuzzy_search.tokenization.token import Tokenizer


def as_phrase_object(phrase: Union[str, dict, Phrase],
                     ngram_size: int = 2,
                     skip_size: int = 2,
                     tokenizer: Tokenizer = None) -> Phrase:
    if isinstance(phrase, Phrase):
        return phrase
    if isinstance(phrase, dict):
        if not is_phrase_dict(phrase):
            print(f"phrase: {phrase}")
            raise KeyError("invalid phrase dictionary")
        return Phrase(phrase, ngram_size=ngram_size, skip_size=skip_size, tokenizer=tokenizer)
    if isinstance(phrase, str):
        return Phrase(phrase, ngram_size=ngram_size, skip_size=skip_size, tokenizer=tokenizer)
    else:
        raise TypeError('phrase must be of type string')


def is_phrase_dict(phrase_dict: Dict[str, Union[str, List[str]]]) -> bool:
    if not isinstance(phrase_dict, dict):
        return False
    if "phrase" not in phrase_dict:
        return False
    if not isinstance(phrase_dict["phrase"], str):
        return False
    if "variants" in phrase_dict:
        for variant in phrase_dict["variants"]:
            if not isinstance(variant, str):
                return False
    if "distractors" in phrase_dict:
        for distractor in phrase_dict["distractors"]:
            if not isinstance(distractor, str):
                return False
    if "labels" in phrase_dict:
        if isinstance(phrase_dict["labels"], str):
            return True
        if not isinstance(phrase_dict["labels"], list):
            return False
        for label in phrase_dict["labels"]:
            if not isinstance(label, str):
                return False
    return True


class PhraseModel:

    def __init__(self, phrases: Union[None, List[Union[str, Dict[str, Union[str, list]], Phrase]]] = None,
                 variants: Union[None, List[Union[Dict[str, List[str]], Phrase]]] = None,
                 phrase_labels: Union[None, List[Dict[str, str]]] = None,
                 distractors: Union[None, List[Union[Dict[str, List[str]], Phrase]]] = None,
                 model: Union[None, List[Dict[str, Union[str, list]]]] = None,
                 custom: Union[None, List[Dict[str, Union[str, int, float, list]]]] = None,
                 config: dict = None,
                 tokenizer: Tokenizer = None):
        if config is None:
            config = {}
        self.ngram_size = config["ngram_size"] if "ngram_size" in config else 2
        self.skip_size = config["skip_size"] if "skip_size" in config else 2
        self.phrase_index: Dict[str, Phrase] = {}
        # only register variants of known phrases
        self.variant_index: Dict[str, Phrase] = {}
        self.has_variants: Dict[str, Set[str]] = defaultdict(set)
        self.is_variant_of: Dict[str, str] = {}
        self.distractor_index: Dict[str, Phrase] = {}
        self.has_distractors: Dict[str, Set[str]] = defaultdict(set)
        self.is_distractor_of: Dict[str, Set[str]] = defaultdict(set)
        self.phrase_length_index: Dict[int, set] = defaultdict(set)
        self.variant_length_index: Dict[int, set] = defaultdict(set)
        self.has_labels: Dict[str, Set[str]] = defaultdict(set)
        self.is_label_of: Dict[str, Set[str]] = defaultdict(set)
        self.custom = {}
        self.word_in_phrase: Dict[str, Set[str]] = defaultdict(set)
        self.token_in_phrase: Dict[str, Set[str]] = defaultdict(set)
        self.first_word_in_phrase: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.first_token_in_phrase: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.min_token_offset_in_phrase: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.max_token_offset_in_phrase: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.phrase_token_max_start_offset: Dict[str, int] = {}
        self.phrase_token_max_end_offset: Dict[str, int] = {}
        self.phrase_type: Dict[str, Set[str]] = defaultdict(set)
        self.phrase_string_map: Dict[str, Phrase] = {}
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        if phrases:
            self.add_phrases(phrases)
        if variants:
            self.add_variants(variants)
        if distractors:
            self.add_distractors(distractors)
        if phrase_labels:
            self.add_labels(phrase_labels)
        if model:
            self.add_model(model)
        if custom:
            self.add_custom(custom)
        self.set_phrase_token_max_start_offsets()
        self.set_phrase_token_max_end_offsets()

    def __repr__(self):
        """A phrase model to support fuzzy searching in OCR/HTR output."""
        return f"PhraseModel({json.dumps(self.json, indent=2)})"

    def __str__(self):
        return self.__repr__()

    def add_model(self, model: List[Union[str, Dict[str, Union[str, list]]]]) -> None:
        """Add an entire model with list of phrase dictionaries.

        :param model: a list of phrase dictionaries
        :type model: List[Union[str, Dict[str, Union[str list]]]]
        :return: None
        :rtype: None
        """
        self.add_phrases(model)
        self.add_variants(model)
        self.add_distractors(model)
        self.add_labels(model)
        self.add_custom(model)

    @property
    def json(self) -> List[Dict[str, Union[str, List[str]]]]:
        """Return a JSON representation of the phrase model.

        :return: a JSON respresentation of the phrase model
        :rtype: List[Dict[str, Union[str, List[str]]]]
        """
        model_json: List[Dict[str, Union[str, List[str]]]] = []
        for phrase in self.phrase_index:
            entry = {'phrase': phrase}
            if phrase in self.has_variants:
                entry['variants'] = list(self.has_variants[phrase])
            if phrase in self.has_labels:
                entry['label'] = list(self.has_labels[phrase])
            if phrase in self.custom:
                entry['custom'] = self.custom[phrase]
            model_json += [entry]
        return model_json

    def add_phrase(self, phrase: Phrase) -> None:
        """Add a phrase to the model as main phrase.

        :param phrase: a phrase to be added
        :type phrase: Phrase
        """
        self.phrase_string_map[phrase.phrase_string] = phrase
        self.phrase_type[phrase.phrase_string].add("phrase")
        self.phrase_index[phrase.phrase_string] = phrase
        self.phrase_length_index[len(phrase.phrase_string)].add(phrase.phrase_string)
        self._index_phrase_words(phrase)
        self._index_phrase_tokens(phrase)

    def add_variant(self, variant_phrase: Phrase, main_phrase: Phrase):
        """Add a phrase to the model as variant of a given main phrase.

        :param variant_phrase: a variant phrase to be added as variant of main_phrase
        :type variant_phrase: Phrase
        :param main_phrase: a main phrase that the variant phrase is a variant of
        :type main_phrase: Phrase
        """
        if variant_phrase.phrase_string not in self.phrase_string_map:
            self.phrase_string_map[variant_phrase.phrase_string] = variant_phrase
        self.variant_index[variant_phrase.phrase_string] = variant_phrase
        self.is_variant_of[variant_phrase.phrase_string] = main_phrase.phrase_string
        self.has_variants[main_phrase.phrase_string].add(variant_phrase.phrase_string)
        self.phrase_type[variant_phrase.phrase_string].add("variant")
        self.variant_length_index[len(variant_phrase.phrase_string)].add(variant_phrase.phrase_string)
        self._index_phrase_words(variant_phrase)
        self._index_phrase_tokens(variant_phrase)

    def add_distractor(self, distractor_phrase: Phrase, main_phrase: Phrase):
        """Add a phrase to the model as distractor of a given main phrase.

        :param distractor_phrase: a distractor phrase to be added as distractor of main_phrase
        :type distractor_phrase: Phrase
        :param main_phrase: a main phrase that the distractor phrase is a distractor of
        :type main_phrase: Phrase
        """
        if distractor_phrase.phrase_string not in self.phrase_string_map:
            self.phrase_string_map[distractor_phrase.phrase_string] = distractor_phrase
        self.distractor_index[distractor_phrase.phrase_string] = distractor_phrase
        self.is_distractor_of[distractor_phrase.phrase_string].add(main_phrase.phrase_string)
        self.has_distractors[main_phrase.phrase_string].add(distractor_phrase.phrase_string)
        self.phrase_type[distractor_phrase.phrase_string].add("distractor")
        self._index_phrase_words(distractor_phrase)
        self._index_phrase_tokens(distractor_phrase)

    def get_phrase(self, phrase_string: str):
        """
        Return the indexed phrase object for a given phrase string,
        or None if no phrase has that phrase string.

        :param phrase_string: a string representation of an indexed phrase
        :type phrase_string: str
        :return: the phrase object that has the given phrase
        :rtype: Union[Phrase, None]
        """
        if phrase_string in self.phrase_index:
            return self.phrase_index[phrase_string]
        elif phrase_string in self.variant_index:
            return self.variant_index[phrase_string]
        elif phrase_string in self.distractor_index:
            return self.distractor_index[phrase_string]
        else:
            return None

    def remove_phrase(self, phrase: Phrase):
        """Remove a main phrase from the model, including its connections to any variant and distractor phrases.

        :param phrase: a phrase that is registered as a main phrase
        :type phrase: Phrase
        """
        # first check if phrase is registered in this phrase model
        if phrase.phrase_string not in self.phrase_index:
            raise ValueError(f"{phrase.phrase_string} is not registered as a main phrase")
        # remove phrase from the type index
        self.phrase_type[phrase.phrase_string].remove("phrase")
        # remove the phrase string from the main phrase index
        del self.phrase_index[phrase.phrase_string]
        # remove the phrase from the phrase length index
        self.phrase_length_index[len(phrase.phrase_string)].remove(phrase.phrase_string)
        if len(self.phrase_type[phrase.phrase_string]) == 0:
            # if the phrase string is not registered as another type (variant or distractor)
            # remove the phrase words from the word_to_phrase index
            self._remove_phrase_words(phrase)
            self._remove_phrase_tokens(phrase)
        # if the phrase has variants, remove those as well
        if phrase.phrase_string in self.has_variants:
            for variant_string in self.has_variants:
                variant_phrase = self.variant_index[variant_string]
                self.remove_variant(variant_phrase)
        # if the phrase has distractors, remove its connections with them as well
        if phrase.phrase_string in self.has_distractors:
            for distractor_string in self.has_distractors:
                distractor_phrase = self.distractor_index[distractor_string]
                if len(self.is_distractor_of[distractor_string]) > 1:
                    self.is_distractor_of[distractor_string].remove(phrase.phrase_string)
                else:
                    # if the distractor is only connected to this phrase, remove the distractor as well
                    self.remove_distractor(distractor_phrase)
            del self.has_distractors[phrase.phrase_string]

    def remove_variant(self, variant_phrase: Phrase) -> None:
        """Remove a variant phrase from the model, including its connection to the phrase it is a
        variant of.

        :param variant_phrase: a phrase that is registered as a variant of one or more main phrases
        :type variant_phrase: Phrase
        """
        # first check if variant phrase is registered as a variant
        if variant_phrase.phrase_string not in self.is_variant_of:
            raise ValueError(f"{variant_phrase.phrase_string} is not registered as a variant")
        # remove variant from the type index
        self.phrase_type[variant_phrase.phrase_string].remove("variant")
        # if that is the only type of the phrase, remove it from the word_to_phrase index
        if len(self.phrase_type[variant_phrase.phrase_string]) == 0:
            self._remove_phrase_words(variant_phrase)
            self._remove_phrase_tokens(variant_phrase)
        # remove the variant from the variant index
        del self.variant_index[variant_phrase.phrase_string]
        # remove the variant from the phrase length index
        self.variant_length_index[len(variant_phrase.phrase_string)].remove(variant_phrase.phrase_string)
        # remove its connection with its main phrase
        main_phrase_string = self.is_variant_of[variant_phrase.phrase_string]
        del self.is_variant_of[variant_phrase.phrase_string]
        self.has_variants[main_phrase_string].remove(variant_phrase.phrase_string)
        # if this was the only variant of the main phrase, remove the main phrase from the has_variants index
        if len(self.has_variants[main_phrase_string]) == 0:
            del self.has_variants[main_phrase_string]

    def remove_distractor(self, distractor_phrase: Phrase) -> None:
        """Remove a distractor phrase from the model, including its connection to the phrase it is a
        distractor of.

        :param distractor_phrase: a phrase that is registered as a distractor of one or more main phrases
        :type distractor_phrase: Phrase
        """
        if distractor_phrase.phrase_string not in self.is_distractor_of:
            raise ValueError(f"{distractor_phrase.phrase_string} is not registered as a distractor")
        self.phrase_type[distractor_phrase.phrase_string].remove("distractor")
        if len(self.phrase_type[distractor_phrase.phrase_string]) == 0:
            self._remove_phrase_words(distractor_phrase)
            self._remove_phrase_tokens(distractor_phrase)
        del self.distractor_index[distractor_phrase.phrase_string]
        for main_phrase_string in self.is_distractor_of[distractor_phrase.phrase_string]:
            self.has_distractors[main_phrase_string].remove(distractor_phrase.phrase_string)
            if len(self.has_distractors[main_phrase_string]) == 0:
                del self.has_distractors[main_phrase_string]
        del self.is_distractor_of[distractor_phrase.phrase_string]

    def add_phrases(self, phrases: List[Union[str, Dict[str, Union[str, List[str]]], Phrase]]) -> None:
        """Add a list of phrases to the phrase model. Phrases must be either:
        - a list of strings
        - a list of dictionaries with property 'phrase' and the phrase as a string value
        - a list of Phrase objects

        :param phrases: a list of phrases
        :type phrases: List[Union[str, Dict[str, Union[str, List[str]]]]]
        """
        for phrase in phrases:
            phrase = as_phrase_object(phrase, ngram_size=self.ngram_size,
                                      skip_size=self.skip_size, tokenizer=self.tokenizer)
            self.add_phrase(phrase)
        # if phrases is a dictionary with possible variants, distractors, labels and custom metadata
        # per phrase, add those variants and distractors
        phrase_dicts = [phrase for phrase in phrases if isinstance(phrase, dict)]
        phrases = [Phrase(phrase_dict, tokenizer=self.tokenizer) for phrase_dict in phrase_dicts]
        self.add_variants(phrases)
        self.add_distractors(phrases)
        self.add_custom(phrases)
        self.add_labels(phrases)

    def remove_phrases(self, phrases: List[Union[str, Dict[str, Union[str, List[str]]], Phrase]]):
        """Remove a list of phrases from the phrase model. If it has any registered spelling variants,
        remove those as well.

        :param phrases: a list of phrases/keyphrases
        :type phrases: List[Union[str, Dict[str, Union[str, List[str]]]]]
        """
        # print('REMOVING PHRASES')
        for phrase in phrases:
            # print('\tphrase:', phrase)
            phrase = as_phrase_object(phrase, ngram_size=self.ngram_size, skip_size=self.skip_size)
            # print('\tas phrase:', phrase)
            if phrase.phrase_string not in self.phrase_index:
                raise KeyError(f"Unknown phrase: {phrase.phrase_string}")
            self.remove_phrase(phrase)

    def get_phrases_by_max_length(self, max_length: int,
                                  include_variants: bool = False) -> Generator[Phrase, None, None]:
        """Return all phrase in the phrase model that are no longer than a given length.

        :param max_length: the maximum length of phrases to be returned
        :type max_length: int
        :param include_variants: whether to include variants
        :return: a generator that yield phrases
        :rtype: Generator[Phrase, None, None]
        """
        for phrase_length in self.phrase_length_index:
            if phrase_length > max_length:
                break
            for phrase_string in self.phrase_length_index[phrase_length]:
                yield self.phrase_index[phrase_string]
            if include_variants:
                for phrase_string in self.variant_length_index[phrase_length]:
                    yield self.variant_index[phrase_string]

    def get_phrases(self) -> List[Phrase]:
        """Return a list of all registered phrases.

        :return: a list of all registered phrases
        :rtype: List[Phrase]
        """
        return list(self.phrase_index.values())

    def has_phrase(self, phrase: Union[str, Dict[str, any], Phrase]) -> bool:
        """Check if phrase is registered in phrase_model.

        :param phrase: a phrase string
        :type phrase: Union[str, Dict[str, any], Phrase]
        :return: a boolean indicating whether phrase is registered
        :rtype: bool
        """
        phrase = as_phrase_object(phrase, ngram_size=self.ngram_size, skip_size=self.skip_size)
        return phrase.phrase_string in self.phrase_index

    def add_variants(self, variants: List[Union[Phrase, Dict[str, Union[str, List[str]]]]],
                     add_new_phrases: bool = True) -> None:
        """Add variants of a phrase. If the phrase is not registered, add it to the set.
        - input is a list of dictionaries:
        variants = [{'phrase': 'some phrase', 'variants': ['some variant', 'some other variant']}]

        :param variants: a list of phrases or phrase dictionaries with 'variant' property
        :type variants: List[Dict[str, Union[str, List[str]]]]
        :param add_new_phrases: a Boolean to indicate if unknown phrases should be added
        :type add_new_phrases: bool
        """
        for phrase in variants:
            main_phrase = as_phrase_object(phrase, ngram_size=self.ngram_size, skip_size=self.skip_size)
            # print(main_phrase.metadata)
            if main_phrase.phrase_string not in self.phrase_index:
                if add_new_phrases:
                    self.add_phrase(main_phrase)
                else:
                    continue
            if "variants" not in main_phrase.metadata:
                continue
            for variant_phrase_string in main_phrase.metadata["variants"]:
                variant_phrase = as_phrase_object(variant_phrase_string, ngram_size=self.ngram_size,
                                                  skip_size=self.skip_size)
                variant_phrase.add_metadata(main_phrase.metadata)
                self.add_variant(variant_phrase, main_phrase)

    def remove_variants(self, variants: Union[List[Union[str, Phrase]], None] = None,
                        variants_of_phrase: Union[str, Phrase, None] = None):
        """Remove a list of spelling variants of a phrase.

        :param variants: a list of variant strings or variant phrase objects to remove
        :type variants: Union[List[str, Phrase]], None]
        :param variants_of_phrase: an optional phrase string or phrase object for which all variants are to be removed
        :type variants_of_phrase: Union[str, Phrase, None]
        """
        if variants:
            for variant in variants:
                variant = as_phrase_object(variant, ngram_size=self.ngram_size, skip_size=self.skip_size)
                self.remove_variant(variant)
        if variants_of_phrase:
            main_phrase = as_phrase_object(variants_of_phrase, ngram_size=self.ngram_size, skip_size=self.skip_size)
            if main_phrase not in self.phrase_index:
                raise IndexError(f"{main_phrase.phrase_string} is not registered in this phrase model")
            if main_phrase not in self.has_variants:
                return None
            for variant_string in self.has_variants[main_phrase.phrase_string]:
                variant = as_phrase_object(variant_string, ngram_size=self.ngram_size, skip_size=self.skip_size)
                self.remove_variant(variant)

    def get_variants(self, phrases: List[str] = None) -> List[Dict[str, Union[str, List[str]]]]:
        """Return registered variants of a specific list of phrases or
        of all registered phrases (when no list of phrases is given).

        :param phrases: a list of registered phrase strings
        :type phrases: List[str]
        :return: a list of dictionaries of phrases and their variants
        :rtype: List[Dict[str, Union[str, List[str]]]]
        """
        if phrases is None:
            phrases = self.phrase_index.keys()
        phrases = [as_phrase_object(phrase, ngram_size=self.ngram_size, skip_size=self.skip_size) for phrase in phrases]
        return [{'phrase': phrase.phrase_string, 'variants': self.has_variants[phrase.phrase_string]}
                for phrase in phrases]

    def variant_of(self, variant: Union[str, Phrase]) -> Union[None, Phrase]:
        variant_string = variant.phrase_string if isinstance(variant, Phrase) else variant
        if variant_string in self.is_variant_of:
            phrase_string = self.is_variant_of[variant_string]
            return self.phrase_index[phrase_string]
        else:
            return None

    def variants(self, phrase: Union[str, Phrase]) -> Union[None, List[Phrase]]:
        """Return all variants of a given phrase.

        :param phrase: a phrase string or phrase object
        :type phrase: Union[str, Phrase]
        :return: a list of variants of the phrase or None if it doesn't have any
        :type: Union[None, List[Phrase]]
        """
        phrase_string = phrase.phrase_string if isinstance(phrase, Phrase) else phrase
        if phrase_string not in self.has_variants:
            return None
        else:
            return [self.variant_index[variant_string] for variant_string in self.has_variants[phrase_string]]

    def add_distractors(self, distractors: List[Union[Phrase, Dict[str, Union[str, List[str]]]]],
                        add_new_phrases: bool = True) -> None:
        """Add distractors of a phrase. If the phrase is not registered, add it to the set.
        - input is a list of dictionaries:
        distractors = [{'phrase': 'some phrase', 'distractors': ['some distractor', 'some other distractor']}]

        :param distractors: a list of phrase dictionaries with 'distractor' property
        :type distractors: List[Dict[str, Union[str, List[str]]]]
        :param add_new_phrases: a Boolean to indicate if unknown phrases should be added
        :type add_new_phrases: bool
        """
        for phrase in distractors:
            main_phrase = as_phrase_object(phrase, ngram_size=self.ngram_size, skip_size=self.skip_size)
            if main_phrase.phrase_string not in self.phrase_index:
                if add_new_phrases:
                    self.add_phrase(main_phrase)
                else:
                    continue
            if "distractors" not in main_phrase.metadata:
                continue
            for distractor_string in main_phrase.metadata["distractors"]:
                distractor_phrase = as_phrase_object(distractor_string, ngram_size=self.ngram_size,
                                                     skip_size=self.skip_size)
                distractor_phrase.add_metadata(main_phrase.metadata)
                self.add_distractor(distractor_phrase, main_phrase)

    def remove_distractors(self, distractors: Union[List[Union[str, Phrase]], None] = None,
                           distractors_of_phrase: Union[str, None] = None):
        """Remove a list of distractors of a phrase.
        - distractors: a list of dictionaries with phrases as key and the list of distractors to be removed as values
        distractors = [{'phrase': 'some phrase', 'distractors': ['distractor to remove', 'some other distractor']}]
        - phrase: remove all distractors of a given phrase

        :param distractors: an optional list of phrase dictionaries with 'distractors' property
        :type distractors: Union[List[Union[str, Phrase]], None]
        :param distractors_of_phrase: an optional string of a registered phrase for which all distractors are removed
        :type distractors_of_phrase: Union[str, None]
        """
        if distractors:
            for distractor in distractors:
                distractor = as_phrase_object(distractor, ngram_size=self.ngram_size, skip_size=self.skip_size)
                self.remove_distractor(distractor)
        if distractors_of_phrase:
            main_phrase = as_phrase_object(distractors_of_phrase, ngram_size=self.ngram_size, skip_size=self.skip_size)
            if main_phrase not in self.phrase_index:
                raise IndexError(f"{main_phrase.phrase_string} is not registered in this phrase model")
            if main_phrase not in self.has_distractors:
                return None
            for distractor_string in self.has_distractors[main_phrase.phrase_string]:
                distractor = as_phrase_object(distractor_string, ngram_size=self.ngram_size, skip_size=self.skip_size)
                self.remove_distractor(distractor)

    def add_labels(self, phrase_labels: List[Union[Phrase, Dict[str, Union[str, List[str]]]]]):
        """Add a label to a phrase. This can be used to group phrases under the same label.
        - input is a list of phrase/label pair dictionaries:
        labels = [{'phrase': 'some phrase', 'label': 'some label'}]
        """
        for phrase in phrase_labels:
            phrase = as_phrase_object(phrase, ngram_size=self.ngram_size, skip_size=self.skip_size)
            if phrase.label is None:
                continue
            if phrase.phrase_string not in self.phrase_index:
                print(f'skipping label for unknown phrase {phrase}')
            for label in phrase.label_set:
                self.has_labels[phrase.phrase_string].add(label)
                self.is_label_of[label].add(phrase.phrase_string)

    def remove_labels(self, phrases: Union[List[Phrase], List[str]]) -> None:
        """Remove labels for known phrases.

        :param phrases: is a list of known phrases (either as Phrase objects or strings)
        :type phrases: Union[List[Phrase], List[str]]
        """
        for phrase in phrases:
            phrase_string = phrase if isinstance(phrase, str) else phrase.phrase_string
            if phrase_string not in self.phrase_index:
                raise TypeError(f'unknown phrase {phrase_string}')
            else:
                for label in self.has_labels[phrase_string]:
                    self.is_label_of[label].remove(phrase_string)
                    if len(self.is_label_of[label]) == 0:
                        del self.is_label_of[label]
                del self.has_labels[phrase_string]

    def is_label(self, label: str) -> bool:
        """Check if label is registered as label of any known phrase.

        :param label: a label string to be checked
        :type label: str
        :return: a boolean whether the label belongs to a known phrase
        :rtype: bool
        """
        return label in self.is_label_of

    def has_label(self, phrase_string: str) -> bool:
        """Check if a registered phrase has a label.

        :param phrase_string: a phrase string of a registered phrase
        :type phrase_string: str
        :return: a boolean indicating if the registered phrase has a label
        """
        return phrase_string in self.has_labels

    def get_labels(self, phrase: Union[str, Phrase]) -> Set[str]:
        """Return the label(s) of a registered phrase.

        :param phrase: a phrase string or object
        :type phrase: Union[str, Phrase]
        :return: a set of labels
        :rtype: List[str]
        """
        phrase_string = phrase if isinstance(phrase, str) else phrase.phrase_string
        if not self.has_phrase(phrase_string):
            raise KeyError(f"Unknown phrase: {phrase}")
        return self.has_labels[phrase]

    def add_custom(self, custom: List[Union[Phrase, Dict[str, Union[str, int, float, list]]]]) -> None:
        """Add custom key/value pairs to the entry as phrase metadata.

        param entry: an Array of phrase dictionaries, each with a 'phrase' property and additional key/value pairs
        type entry: Dict[str, Union[str, int, float, list]]
        """
        for entry in custom:
            phrase = as_phrase_object(entry, ngram_size=self.ngram_size, skip_size=self.skip_size)
            if phrase.metadata is None:
                continue
            if phrase.phrase_string not in self.phrase_index:
                continue
            self.custom[phrase.phrase_string] = copy.deepcopy(phrase.metadata)

    def remove_custom(self, custom: List[Dict[str, any]]) -> None:
        """Remove custom properties for a list of phrases.

        :param custom: a list of phrase dictionaries with custom properties to remove
        :type custom: List[Dict[str, any]]
        """
        for entry in custom:
            phrase = as_phrase_object(entry, ngram_size=self.ngram_size, skip_size=self.skip_size)
            for custom_property in phrase.metadata:
                del self.custom[phrase.phrase_string][custom_property]

    def has_custom(self, phrase_string: str, custom_property: str) -> bool:
        """Check if a phrase has a given custom property.

        :param phrase_string: a phrase string of a registered phrase.
        :type phrase_string: str
        :param custom_property: the name of a custom property of the registered phrase
        :type custom_property: str
        :return: a boolean to indicate whether the phrase has a custom property of the given property name
        :rtype: bool
        """
        # print('CUSTOM:', self.custom)
        return phrase_string in self.custom and custom_property in self.custom[phrase_string]

    def get(self, phrase_string: str, custom_property: str) -> any:
        """Get the value of a custom property for a given phrase.

        :param phrase_string: a phrase string of a registered phrase.
        :type phrase_string: str
        :param custom_property: the name of a custom property of the registered phrase
        :type custom_property: str
        :return: the custom property of a given phrase
        :rtype: any
        """
        if phrase_string not in self.phrase_index:
            raise KeyError("Unknown phrase_string")
        if not self.has_custom(phrase_string, custom_property):
            raise ValueError("Unknown custom property")
        return self.custom[phrase_string][custom_property]

    def _index_phrase_words(self, phrase: Phrase) -> None:
        """Index a phrase on its individual words, for exact match look up routines.

        :param phrase: a phrase object that is part of the phrase model
        :type phrase: Phrase
        """
        if phrase.phrase_string not in self.phrase_type:
            raise ValueError(f"Cannot index phrase words for non-registered phrase: {phrase.phrase_string}")
        for wi, word in enumerate(re.finditer(r"\w+", phrase.phrase_string)):
            if wi == 0:
                self.first_word_in_phrase[word.group(0)][phrase.phrase_string] = word.start()
            self.word_in_phrase[word.group(0)].add(phrase.phrase_string)

    def _remove_phrase_words(self, phrase: Phrase) -> None:
        """Remove the individual words of a phrase from the index. Only use this is you are removing
        the phrase from the phrase model.

        :param phrase: a phrase object that is part of the phrase model
        :type phrase: Phrase
        """
        for wi, word in enumerate(re.finditer(r"\w+", phrase.phrase_string)):
            if wi == 0:
                del self.first_word_in_phrase[word.group(0)][phrase.phrase_string]
                if len(self.first_word_in_phrase[word.group(0)].keys()) == 0:
                    del self.first_word_in_phrase[word.group(0)]
            self.word_in_phrase[word.group(0)].remove(phrase.phrase_string)
            if len(self.word_in_phrase[word.group(0)]) == 0:
                del self.word_in_phrase[word.group(0)]

    def _remove_phrase_tokens(self, phrase: Phrase, tokenizer: Tokenizer = None) -> None:
        """Remove the individual words of a phrase from the index. Only use this is you are removing
        the phrase from the phrase model.

        :param phrase: a phrase object that is part of the phrase model
        :type phrase: Phrase
        """
        tokenizer = self._get_tokenizer(tokenizer)
        if tokenizer is None:
            return None
        tokens = tokenizer.tokenize(phrase.phrase_string)
        for ti, token in enumerate(tokens):
            if ti == 0:
                if token.n not in self.first_token_in_phrase:
                    print(f"phrase_model._remove_phrase_tokens - token not in first_token_in_phrase index")
                    print(f"    token: {token.n}")
                    print(f"    first_token_in_phrase.keys(): {self.first_token_in_phrase.keys()}")
                del self.first_token_in_phrase[token.n][phrase.phrase_string]
                if len(self.first_token_in_phrase[token.n].keys()) == 0:
                    del self.first_token_in_phrase[token.n]
            self.token_in_phrase[token.n].remove(phrase.phrase_string)
            del self.min_token_offset_in_phrase[token.n][phrase.phrase_string]
            del self.max_token_offset_in_phrase[token.n][phrase.phrase_string]
            if len(self.token_in_phrase[token.n]) == 0:
                del self.token_in_phrase[token.n]
            if len(self.min_token_offset_in_phrase[token.n]) == 0:
                del self.min_token_offset_in_phrase[token.n]
                del self.max_token_offset_in_phrase[token.n]

    def _get_tokenizer(self, tokenizer: Tokenizer = None):
        return tokenizer if tokenizer is not None else self.tokenizer

    def _index_phrase_tokens(self, phrase: Phrase, tokenizer: Tokenizer = None):
        tokenizer = self._get_tokenizer(tokenizer)
        if tokenizer:
            phrase.tokens = tokenizer.tokenize(phrase.phrase_string, doc_id=phrase.phrase_string)
            for ti, token in enumerate(phrase.tokens):
                if ti == 0:
                    self.first_token_in_phrase[token.n][phrase.phrase_string] = token.char_index
                if token.n not in self.min_token_offset_in_phrase or \
                        phrase.phrase_string not in self.min_token_offset_in_phrase[token.n]:
                    self.min_token_offset_in_phrase[token.n][phrase.phrase_string] = token.char_index
                self.max_token_offset_in_phrase[token.n][phrase.phrase_string] = token.char_index
                self.token_in_phrase[token.n].add(phrase.phrase_string)

    def has_token(self, token: Union[str, Token]):
        return token.n in self.token_in_phrase

    def set_phrase_token_max_start_offsets(self):
        """Check if a token only occurs in phrases with a max start offset, and if so
        set its max."""
        for token in self.token_in_phrase:
            token_has_phrase_with_max_start = False
            token_has_phrase_without_max_start = False
            max_start = -1
            for phrase_string in self.token_in_phrase[token]:
                phrase = self.get_phrase(phrase_string)
                if isinstance(phrase, Phrase) and phrase.has_max_start_offset() is True:
                    token_has_phrase_with_max_start = True
                    # print(f"phrase_model.PhraseModel.set_phrase_token_max_start_offset:")
                    # print(f"    token: {token}\tphrase: {phrase}")
                    # print(f"    phrase.max_start_offset: {phrase.max_start_offset}")
                    # print(f"    self.max_token_offset_in_phrase[token][phrase_string]: "
                    #       f"{self.max_token_offset_in_phrase[token][phrase_string]}")
                    token_max_offset = phrase.max_start_offset + self.max_token_offset_in_phrase[token][phrase_string]
                    # print(f"    token_max_offset: {token_max_offset}")
                    if token_max_offset > max_start:
                        max_start = token_max_offset
                    # print(f"    max_start: {max_start}")
                if isinstance(phrase, Phrase) and phrase.has_max_start_offset() is False:
                    token_has_phrase_without_max_start = True
                    # print(f"without max_start - token: {token}\tphrase: {phrase_string}")
            if token_has_phrase_with_max_start and not token_has_phrase_without_max_start:
                self.phrase_token_max_start_offset[token] = max_start

    def set_phrase_token_max_end_offsets(self):
        """Check if a token only occurs in phrases with a max end offset, and if so
        set its max."""
        for token in self.token_in_phrase:
            token_has_phrase_with_max_end = False
            token_has_phrase_without_max_end = False
            max_end = -1
            for phrase_string in self.token_in_phrase[token]:
                phrase = self.get_phrase(phrase_string)
                if isinstance(phrase, Phrase) and phrase.has_max_end_offset() is True:
                    token_has_phrase_with_max_end = True
                    token_max_offset = phrase.max_end_offset + self.min_token_offset_in_phrase[token][phrase_string]
                    if token_max_offset > max_end:
                        max_end = token_max_offset
                if isinstance(phrase, Phrase) and phrase.has_max_end_offset() is False:
                    token_has_phrase_without_max_end = True
            if token_has_phrase_with_max_end and not token_has_phrase_without_max_end:
                self.phrase_token_max_end_offset[token] = max_end
