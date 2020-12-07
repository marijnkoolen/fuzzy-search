from typing import Dict, List, Set, Union
from collections import defaultdict
import json
import copy

from fuzzy_search.fuzzy_phrase import Phrase


class PhraseModel:

    def __init__(self, phrases: Union[None, List[Union[str, Dict[str, Union[str, list]], Phrase]]] = None,
                 variants: Union[None, List[Union[Dict[str, List[str]], Phrase]]] = None,
                 phrase_labels: Union[None, List[Dict[str, str]]] = None,
                 distractors: Union[None, List[Union[Dict[str, List[str]], Phrase]]] = None,
                 model: Union[None, List[Dict[str, Union[str, list]]]] = None,
                 custom: Union[None, List[Dict[str, Union[str, int, float, list]]]] = None):
        self.phrase_index: Dict[str, Phrase] = {}
        # only register variants of known phrases
        self.variant_index: Dict[str, Phrase] = {}
        self.has_variants: Dict[str, Set[str]] = defaultdict(set)
        self.is_variant_of: Dict[str, str] = {}
        self.distractor_index: Dict[str, Phrase] = {}
        self.has_distractors: Dict[str, Set[str]] = defaultdict(set)
        self.is_distractor_of: Dict[str, List[str]] = {}
        self.labels = {}
        self.custom = {}
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

    def __repr__(self):
        """A phrase model to support fuzzy searching in OCR/HTR output."""
        return f"PhraseModel({json.dumps(self.to_json(), indent=2)})"

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

    def to_json(self) -> List[Dict[str, Union[str, List[str]]]]:
        """Return a JSON representation of the phrase model.

        :return: a JSON respresentation of the phrase model
        :rtype: List[Dict[str, Union[str, List[str]]]]
        """
        model_json: List[Dict[str, Union[str, List[str]]]] = []
        for phrase in self.phrase_index:
            entry = {'phrase': self.phrase_index[phrase].metadata}
            if phrase in self.has_variants:
                entry['variants'] = self.has_variants[phrase]
            if phrase in self.labels:
                entry['label'] = self.labels[phrase]
            model_json += [entry]
        return model_json

    def add_phrases(self, phrases: List[Union[str, Dict[str, Union[str, List[str]]], Phrase]]) -> None:
        """Add a list of phrases to the phrase model. Keywords must be either:
        - a list of strings
        - a list of dictionaries with property 'phrase' and the phrase as a string value
        - a list of Phrase objects

        :param phrases: a list of phrases
        :type phrases: List[Union[str, Dict[str, Union[str, List[str]]]]]
        """
        for phrase in phrases:
            if isinstance(phrase, Phrase):
                continue
            if isinstance(phrase, dict):
                if 'phrase' not in phrase:
                    raise KeyError("Keywords as list of dictionaries should have 'phrase' property")
                if not isinstance(phrase['phrase'], str):
                    raise TypeError('phrases mut be of type string')
            elif not isinstance(phrase, str):
                raise TypeError('phrases mut be of type string')
        for phrase in phrases:
            if not isinstance(phrase, Phrase):
                phrase = Phrase(phrase)
            self.phrase_index[phrase.phrase_string] = phrase
        phrase_dicts = [phrase for phrase in phrases if isinstance(phrase, dict)]
        self.add_variants(phrase_dicts)
        self.add_distractors(phrase_dicts)
        self.add_custom(phrase_dicts)
        self.add_labels(phrase_dicts)

    def remove_phrases(self, phrases: List[Union[str, Dict[str, Union[str, List[str]]], Phrase]]):
        """Remove a list of phrases from the phrase model. If it has any registered spelling variants,
        remove those as well.

        :param phrases: a list of phrases/keyphrases
        :type phrases: List[Union[str, Dict[str, Union[str, List[str]]]]]
        """
        for phrase in phrases:
            if isinstance(phrase, dict):
                if 'phrase' not in phrase:
                    raise KeyError("Keywords as list of dictionaries should have 'phrase' property")
                else:
                    phrase_string = phrase['phrase']
            elif isinstance(phrase, str):
                phrase_string = phrase
            else:
                phrase_string = phrase.phrase_string
            if phrase_string not in self.phrase_index:
                raise KeyError(f"Unknown phrase: {phrase}")
            del self.phrase_index[phrase_string]
            if phrase_string in self.has_variants:
                del self.has_variants[phrase_string]

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
        if isinstance(phrase, str):
            phrase_string = phrase
        elif isinstance(phrase, dict):
            phrase_string = phrase["phrase"]
        else:
            phrase_string = phrase.phrase_string
        return phrase_string in self.phrase_index

    def add_variants(self, variants: List[Dict[str, Union[str, List[str]]]],
                     add_new_phrases: bool = True) -> None:
        """Add variants of a phrase. If the phrase is not registered, add it to the set.
        - input is a list of dictionaries:
        variants = [
            {'phrase': 'some phrase', 'variants': ['some variant', 'some other variant']}
        ]

        :param variants: a list of phrase dictionaries with 'variant' property
        :type variants: List[Dict[str, Union[str, List[str]]]]
        :param add_new_phrases: a Boolean to indicate if unknown phrases should be added
        :type add_new_phrases: bool
        """
        # first, check that all variants of all phrases are strings
        for phrase_variants in variants:
            if not isinstance(phrase_variants["phrase"], str):
                raise TypeError("phrases must be of type string")
            if "variants" not in phrase_variants:
                continue
            for variant in phrase_variants["variants"]:
                if not isinstance(variant, str):
                    raise TypeError("spelling variants must be of type string")
        for phrase_variants in variants:
            if phrase_variants["phrase"] not in self.phrase_index and add_new_phrases:
                self.add_phrases(phrase_variants["phrase"])
            elif phrase_variants["phrase"] not in self.phrase_index:
                continue
            if "variants" not in phrase_variants:
                continue
            # make sure the list variants is a copy of the original and not a reference to the same list
            variant_phrases = [Phrase(variant_phrase) for variant_phrase in phrase_variants["variants"]]
            for variant in variant_phrases:
                self.variant_index[variant.phrase_string] = variant
                self.is_variant_of[variant.phrase_string] = phrase_variants["phrase"]
                self.has_variants[phrase_variants["phrase"]].add(variant.phrase_string)

    def remove_variants(self, variants: Union[List[Union[str, Phrase]], None] = None,
                        phrase: Union[str, Phrase, None] = None):
        """Remove a list of spelling variants of a phrase.

        :param variants: a list of variant strings or variant phrase objects to remove
        :type variants: Union[List[str, Phrase]], None]
        :param phrase: an optional phrase string or phrase object for which all variants are to be removed
        :type phrase: Union[str, Phrase, None]
        """
        if variants:
            variant_strings = [variant.phrase_string
                               if isinstance(variant, Phrase) else variant for variant in variants]
            for variant_string in variant_strings:
                if variant_string not in self.variant_index:
                    continue
                del self.variant_index[variant_string]
                phrase_string = self.is_variant_of[variant_string]
                del self.is_variant_of[variant_string]
                self.has_variants[phrase_string].remove(variant_string)
        if phrase is not None:
            phrase_string = phrase.phrase_string if isinstance(phrase, Phrase) else phrase
            if phrase_string in self.has_variants:
                del self.has_variants[phrase_string]
            elif phrase_string:
                print('Unknown phrase:', phrase_string)

    def get_variants(self, phrases: List[str] = None) -> List[Dict[str, Union[str, List[str]]]]:
        """Return registered variants of a specific list of phrases or
        of all registered phrases (when no list of phrases is given).

        :param phrases: a list of registered phrase strings
        :type phrases: List[str]
        :return: a list of dictionaries of phrases and their variants
        :rtype: List[Dict[str, Union[str, List[str]]]]
        """
        if phrases:
            for phrase in phrases:
                if not isinstance(phrase, str):
                    raise ValueError('Keywords must be of type string')
                if phrase not in self.phrase_index:
                    raise ValueError('Unknown phrase')
            return [{'phrase': phrase, 'variants': self.has_variants[phrase]} for phrase in phrases]
        else:
            # return variants of all registered phrases
            return [{'phrase': phrase, 'variants': self.has_variants[phrase]} for phrase in self.has_variants]

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

    def add_distractors(self, distractors: List[Dict[str, Union[str, List[str]]]],
                        add_new_phrases: bool = True) -> None:
        """Add distractors of a phrase. If the phrase is not registered, add it to the set.
        - input is a list of dictionaries:
        distractors = [
            {'phrase': 'some phrase', 'distractors': ['some distractor', 'some other distractor']}
        ]

        :param distractors: a list of phrase dictionaries with 'distractor' property
        :type distractors: List[Dict[str, Union[str, List[str]]]]
        :param add_new_phrases: a Boolean to indicate if unknown phrases should be added
        :type add_new_phrases: bool
        """
        # first, check that all distractors of all phrases are strings
        for phrase_distractors in distractors:
            if not isinstance(phrase_distractors["phrase"], str):
                raise TypeError("phrases must be of type string")
            if "distractors" not in phrase_distractors:
                continue
            for distractor in phrase_distractors["distractors"]:
                if not isinstance(distractor, str):
                    raise TypeError("distractors must be of type string")
        for phrase_distractors in distractors:
            phrase_string = phrase_distractors["phrase"]
            if phrase_string not in self.phrase_index and add_new_phrases:
                self.add_phrases([phrase_distractors])
            elif phrase_string not in self.phrase_index:
                continue
            if "distractors" not in phrase_distractors:
                continue
            # make sure the list distractors is a copy of the original and not a reference to the same list
            distractors = [Phrase(distractor_string) for distractor_string in phrase_distractors["distractors"]]
            for distractor in distractors:
                self.distractor_index[distractor.phrase_string] = distractor
                self.is_distractor_of[distractor.phrase_string] = phrase_string
                self.has_distractors[phrase_string].add(distractor.phrase_string)

    def remove_distractors(self, distractors: Union[List[Dict[str, List[str]]], None] = None,
                           phrase: Union[str, None] = None):
        """Remove a list of distractors of a phrase.
        - distractors: a list of dictionaries with phrases as key and the list of distractors to be removed as values
        distractors = [
            {'phrase': 'some phrase', 'distractors': ['some distractor', 'some other distractor']}
        ]
        - phrase: remove all distractors of a given phrase

        :param distractors: an optional list of phrase dictionaries with 'distractors' property
        :type distractors: Union[List[Dict[str, Union[str, List[str]]]], None]
        :param phrase: an optional string of a registered phrase for which all distractors are to be removed
        :type phrase: Union[str, None]
        """
        if distractors:
            for phrase_distractors in distractors:
                if phrase_distractors['phrase'] not in distractors:
                    raise KeyError(f"Cannot remove distractors of unknown phrase {phrase_distractors['phrase']}")
                for distractor in phrase_distractors['distractors']:
                    if distractor in self.has_distractors[phrase]:
                        self.has_distractors[phrase].remove(distractor)
        if phrase and phrase in self.has_distractors:
            del self.has_distractors[phrase]
        elif phrase:
            print('Unknown phrase:', phrase)

    def get_labels(self, phrases: List[str] = None) -> List[Dict[str, str]]:
        """Return a list of phrases and their labels, either for a given list of phrases or
        for all registered phrases."""
        if phrases:
            for phrase in phrases:
                if not isinstance(phrase, str):
                    raise ValueError('Keywords must be of type string')
                if phrase not in self.phrase_index:
                    raise ValueError('Unknown phrase')
            return [{'phrase': phrase, 'label': self.labels[phrase]} for phrase in phrases]
        else:
            # return variants of all registered phrases
            return [{'phrase': phrase, 'label': self.labels[phrase]} for phrase in self.phrase_index]

    def add_labels(self, phrase_labels: List[Dict[str, Union[str, list]]]):
        """Add a label to a phrase. This can be used to group phrases under the same label.
        - input is a list of phrase/label pair dictionaries:
        labels = [
            {'phrase': 'some phrase', 'label': 'some label'}
        ]
        """
        for phrase_label in phrase_labels:
            if 'label' not in phrase_label:
                continue
            if not isinstance(phrase_label['label'], str):
                raise TypeError('phrase labels must be of type string')
        for phrase_label in phrase_labels:
            if 'label' not in phrase_label:
                continue
            phrase = phrase_label['phrase']
            label = phrase_label['label']
            if phrase not in self.phrase_index:
                print(f'skipping label for unknown phrase {phrase}')
            self.labels[phrase] = label

    def remove_labels(self, phrases: List[str]):
        """Remove labels for known phrases. Input is a list of known phrases"""
        for phrase in phrases:
            if phrase not in self.phrase_index:
                raise TypeError(f'unknown phrase {phrase}')
            else:
                del self.labels[phrase]

    def has_label(self, phrase: str) -> bool:
        """Check if a registered phrase has a label."""
        return phrase in self.labels

    def get_label(self, phrase: str) -> str:
        """Return the label of a registered phrase"""
        if phrase not in self.labels:
            raise KeyError(f"Unknown phrase: {phrase}")
        return self.labels[phrase]

    def check_entry_phrase(self, entry: Dict[str, Union[str, int, float, list]]):
        """Check if a given phrase (as dictionary) is registered.

        param entry: a phrase dictionary with a 'phrase' property
        type entry: Dict[str, Union[str, int, float, list]]
        """
        if 'phrase' not in entry:
            raise KeyError("Keywords as list of dictionaries should have 'phrase' property")
        if not isinstance(entry['phrase'], str):
            raise ValueError("phrase property must be a string")
        if entry['phrase'] not in self.phrase_index:
            raise KeyError("Unknown phrase")

    def add_custom(self, custom: List[Dict[str, Union[str, int, float, list]]]):
        """Add custom key/value pairs to the entry as phrase metadata.

        param entry: an Array of phrase dictionaries, each with a 'phrase' property and additional key/value pairs
        type entry: Dict[str, Union[str, int, float, list]]
        """
        for entry in custom:
            self.check_entry_phrase(entry)
            # make sure the custom entry is a copy of the original and not a reference to the same object
            self.custom[entry['phrase']] = copy.copy(entry)

    def remove_custom(self, custom: List[Dict[str, Union[str, int, float, list]]]):
        """Remove custom properties for a list of phrases"""
        for entry in custom:
            self.check_entry_phrase(entry)
            for custom_property in entry:
                del self.custom[entry['phrase']][custom_property]

    def has_custom(self, phrase: str, custom_property: str) -> bool:
        """Check if a phrase has a given custom property."""
        return phrase in self.custom and custom_property in self.custom[phrase]

    def get(self, phrase: str, custom_property: str) -> Union[str, int, float, list]:
        """Get the value of a custom property for a given phrase."""
        if phrase not in self.phrase_index:
            raise KeyError("Unknown phrase")
        if not self.has_custom(phrase, custom_property):
            raise ValueError("Unknown custom property")
        return self.custom[phrase][custom_property]
