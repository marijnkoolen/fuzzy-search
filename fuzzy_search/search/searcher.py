import copy
import string
from collections import defaultdict
from typing import Dict, List, Set, Union

import fuzzy_search
from fuzzy_search.match.skip_match import SkipMatches
from fuzzy_search.match.phrase_match import PhraseMatch
from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.phrase.phrase_model import PhraseModel
from fuzzy_search.search.config import default_config
from fuzzy_search.tokenization.string import text2skipgrams
from fuzzy_search.tokenization.token import Tokenizer


class FuzzySearcher(object):

    def __init__(self, phrase_list: List[any] = None, phrase_model: Union[Dict[str, any], PhraseModel] = None,
                 config: Union[None, Dict[str, Union[str, int, float]]] = None, tokenizer: Tokenizer = None):
        """This class represents the basic fuzzy searcher. You can pass a list of phrases or a phrase model and
        configuration dictionary that overrides the default configuration values. The default config dictionary
        is available via `fuzzy_search.default_config`.

        To set e.g. the character ngram_size to 3 and the skip_size to 1 use the following dictionary:

        config = {
            'ngram_size': 3,
            'skip_size': 1
        }

        :param phrase_list: a list of phrases (a list of strings or more complex dictionaries with phrases and variants)
        :type phrase_list: list
        :param phrase_model: a phrase model
        :type phrase_model: PhraseModel
        :param config: a configuration dictionary to override default configuration properties.
        Only the properties in the config dictionaries of updated.
        :type config: dict
        :param tokenizer: a tokenizer instance (default tokenizer splits on whitespace)
        :type tokenizer: Tokenizer
        """
        self.__version__ = fuzzy_search.__version__
        # default configuration
        self.char_match_threshold = 0.5
        self.ngram_threshold = 0.5
        self.skipgram_threshold = 0.2
        self.levenshtein_threshold = 0.5
        self.max_length_variance = 1
        self.allow_overlapping_matches = False
        self.skip_exact_matching = False
        self.use_word_boundaries = True
        self.ignorecase = False
        self.known_candidates = defaultdict(dict)
        self.distractor_terms = defaultdict(list)
        self.ngram_size = 2
        self.skipgram_index = defaultdict(set)
        self.early_skipgram_index = defaultdict(set)
        self.late_skipgram_index = defaultdict(set)
        self.skip_size = 2
        self.variant_map = defaultdict(dict)
        self.has_variant = defaultdict(dict)
        self.variant_skipgram_index = defaultdict(set)
        self.variant_early_skipgram_index = defaultdict(set)
        self.variant_late_skipgram_index = defaultdict(set)
        self.include_variants = False
        self.filter_distractors = False
        self.distractor_map = defaultdict(dict)
        self.has_distractor = defaultdict(dict)
        self.distractor_skipgram_index = defaultdict(set)
        self.distractor_early_skipgram_index = defaultdict(set)
        self.distractor_late_skipgram_index = defaultdict(set)
        self.phrases: Set[Phrase] = set()
        self.variants: Set[Phrase] = set()
        self.distractors: Set[Phrase] = set()
        self.phrase_model: Union[None, PhraseModel] = None
        self.debug = False
        self.punctuation = string.punctuation
        # non-default configuration
        self.config = copy.deepcopy(default_config)
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        if config:
            for key in config:
                self.config[key] = config[key]
            self.configure(config)
        if phrase_list is not None:
            phrase_model = PhraseModel(phrases=phrase_list, config=config, tokenizer=self.tokenizer)
            self.index_phrase_model(phrase_model)
        if phrase_model is not None:
            if isinstance(phrase_model, dict) or isinstance(phrase_model, list):
                phrase_model = PhraseModel(model=phrase_model, config=config, tokenizer=self.tokenizer)
            elif isinstance(phrase_model, PhraseModel) is False:
                raise TypeError('invalid phrase_model type, should PhraseModel or a list of dictionaries')
            self.index_phrase_model(phrase_model)

    def configure(self, config: Dict[str, any]) -> None:
        """Configure the fuzzy searcher with a given config object.

        :param config: a config dictionary
        :type config: Dict[str, Union[str, int, float]]
        """
        if "char_match_threshold" in config:
            self.char_match_threshold = config["char_match_threshold"]
        if "ngram_threshold" in config:
            self.ngram_threshold = config["ngram_threshold"]
        if "skipgram_threshold" in config:
            self.skipgram_threshold = config["skipgram_threshold"]
        if "levenshtein_threshold" in config:
            self.levenshtein_threshold = config["levenshtein_threshold"]
        if "max_length_variance" in config:
            self.max_length_variance = config["max_length_variance"]
        if "use_word_boundaries" in config:
            self.use_word_boundaries = config["use_word_boundaries"]
        if "ignorecase" in config:
            self.ignorecase = config["ignorecase"]
        if "ngram_size" in config:
            self.ngram_size = config["ngram_size"]
        if "skip_size" in config:
            self.skip_size = config["skip_size"]
        if "include_variants" in config:
            self.include_variants = config["include_variants"]
        if "filter_distractors" in config:
            self.filter_distractors = config["filter_distractors"]
        if "skip_exact_matching" in config:
            self.skip_exact_matching = config["skip_exact_matching"]
        if "allow_overlapping_matches" in config:
            self.allow_overlapping_matches = config["allow_overlapping_matches"]
        if "punctuation" in config:
            self.punctuation = config["punctuation"]
        if "debug" in config:
            self.debug = config["debug"]

    def _get_debug_level(self, debug: int = 0):
        if debug > self.debug:
            return debug
        else:
            return self.debug

    def index_phrase_model(self, phrase_model: Union[List[Dict[str, Union[str, int, float, list]]], PhraseModel],
                           debug: int = 0):
        """Add a phrase model to search for phrases in texts.

        :param phrase_model: a phrase model, either as dictionary or as PhraseModel object
        :type phrase_model: Union[List[Dict[str, Union[str, int, float, list]]], PhraseModel]
        :param debug: level to show debug information
        :type debug: int
        """
        debug = self._get_debug_level(debug)
        if isinstance(phrase_model, list):
            phrase_model = PhraseModel(model=phrase_model, config=self.config, tokenizer=self.tokenizer)
        self.phrase_model = phrase_model
        if debug > 3:
            print(f'{self.__class__.__name__}.index_phrase_model - calling index_phrases()')
        self.index_phrases(list(phrase_model.phrase_index.values()))
        if debug > 3:
            print(f'{self.__class__.__name__}.index_phrase_model - calling index_variants()')
        self.index_variants(list(phrase_model.variant_index.values()))
        if debug > 3:
            print(f'{self.__class__.__name__}.index_phrase_model - calling index_distractors()')
        self.index_distractors(list(phrase_model.distractor_index.values()))

    def index_phrases(self, phrases: List[Union[str, Phrase]]) -> None:
        """Add a list of phrases to search for in texts.

        :param phrases: a list of phrases, either as string or as Phrase objects
        :type phrases: List[Union[str, Phrase]]
        """
        for phrase in phrases:
            if isinstance(phrase, str):
                phrase = Phrase(phrase, ngram_size=self.ngram_size, skip_size=self.skip_size)
            if phrase.ngram_size != self.ngram_size:
                searcher_size = f"{self.__class__.__name__} ({self.ngram_size}"
                raise ValueError(f"phrase has different ngram_size ({phrase.ngram_size}) than {searcher_size}")
            if phrase.skip_size != self.skip_size:
                searcher_size = f"{self.__class__.__name__} ({self.skip_size}"
                raise ValueError(f"phrase has different skip_size ({phrase.skip_size}) than {searcher_size}")
            self.phrases.add(phrase)
            if self.ignorecase:
                # print(f'indexing phrase {phrase.phrase_string} with lowercase')
                for skipgram in phrase.skipgrams_lower:
                    self.skipgram_index[skipgram.string].add(phrase)
                for skipgram_string in phrase.early_skipgram_index_lower:
                    # print('early skipgram_string:', skipgram_string)
                    self.early_skipgram_index[skipgram_string].add(phrase)
                for skipgram_string in phrase.late_skipgram_index_lower:
                    self.late_skipgram_index[skipgram_string].add(phrase)
            else:
                for skipgram in phrase.skipgrams:
                    self.skipgram_index[skipgram.string].add(phrase)
                for skipgram_string in phrase.early_skipgram_index:
                    self.early_skipgram_index[skipgram_string].add(phrase)
                for skipgram_string in phrase.late_skipgram_index:
                    self.late_skipgram_index[skipgram_string].add(phrase)
        if self.phrase_model is None:
            self.phrase_model = PhraseModel(phrases=list(self.phrases))

    def index_variants(self, variants: List[Union[str, Phrase]]) -> None:
        """Add a list of variant phrases to search for in texts.

        :param variants: a list of variants, either as string or as Phrase objects
        :type variants: List[Union[str, Phrase]]
        """
        for variant in variants:
            if isinstance(variant, str):
                variant = Phrase(variant, ngram_size=self.ngram_size, skip_size=self.skip_size)
            if variant.ngram_size != self.ngram_size:
                searcher_size = f"{self.__class__.__name__} ({self.ngram_size}"
                raise ValueError(f"variant has different ngram_size ({variant.ngram_size}) than {searcher_size}")
            if variant.skip_size != self.skip_size:
                searcher_size = f"{self.__class__.__name__} ({self.skip_size}"
                raise ValueError(f"variant has different skip_size ({variant.skip_size}) than {searcher_size}")
            self.variants.add(variant)
            if self.ignorecase:
                for skipgram in variant.skipgrams_lower:
                    self.variant_skipgram_index[skipgram.string].add(variant)
                for skipgram_string in variant.early_skipgram_index_lower:
                    self.variant_early_skipgram_index[skipgram_string].add(variant)
                for skipgram_string in variant.late_skipgram_index_lower:
                    self.variant_late_skipgram_index[skipgram_string].add(variant)
            else:
                for skipgram in variant.skipgrams:
                    self.variant_skipgram_index[skipgram.string].add(variant)
                for skipgram_string in variant.early_skipgram_index:
                    self.variant_early_skipgram_index[skipgram_string].add(variant)
                for skipgram_string in variant.late_skipgram_index:
                    self.variant_late_skipgram_index[skipgram_string].add(variant)

    def index_distractors(self, distractors: List[Union[str, Phrase]]) -> None:
        """Add a list of distractor phrases to filter out likely incorrect phrase matches.

        :param distractors: a list of distractors, either as string or as Phrase objects
        :type distractors: List[Union[str, Phrase]]
        """
        for distractor in distractors:
            if isinstance(distractor, str):
                distractor = Phrase(distractor, ngram_size=self.ngram_size, skip_size=self.skip_size)
            if distractor.ngram_size != self.ngram_size:
                searcher_size = f"{self.__class__.__name__} ({self.ngram_size}"
                raise ValueError(f"distractor has different ngram_size ({distractor.ngram_size}) than {searcher_size}")
            if distractor.skip_size != self.skip_size:
                searcher_size = f"{self.__class__.__name__} ({self.skip_size}"
                raise ValueError(f"distractor has different skip_size ({distractor.skip_size}) than {searcher_size}")
            self.distractors.add(distractor)
            if self.ignorecase:
                for skipgram in distractor.skipgrams_lower:
                    self.distractor_skipgram_index[skipgram.string].add(distractor)
                for skipgram_string in distractor.early_skipgram_index_lower:
                    self.distractor_early_skipgram_index[skipgram_string].add(distractor)
                for skipgram_string in distractor.late_skipgram_index_lower:
                    self.distractor_late_skipgram_index[skipgram_string].add(distractor)
            else:
                for skipgram in distractor.skipgrams:
                    self.distractor_skipgram_index[skipgram.string].add(distractor)
                for skipgram_string in distractor.early_skipgram_index:
                    self.distractor_early_skipgram_index[skipgram_string].add(distractor)
                for skipgram_string in distractor.late_skipgram_index:
                    self.distractor_late_skipgram_index[skipgram_string].add(distractor)

    def find_skipgram_matches(self, text: Dict[str, Union[str, int, float, list]],
                              include_variants: Union[None, bool] = None,
                              known_word_start_offset: Dict[int, Dict[str, any]] = None) -> SkipMatches:
        """Find all skipgram matches between text and phrases.

        :param text: the text object to match with phrases
        :type text: Dict[str, Union[str, int, float, list]]
        :param include_variants: boolean flag for whether to include phrase variants for finding matches
        :type include_variants: bool
        :param known_word_start_offset: a dictionary of known words and their text start_offsets based on exact matches
        :type known_word_start_offset: Dict[int, Dict[str, any]]
        :return: a SkipMatches object contain all skipgram matches
        :rtype: SkipMatches
        """
        # print(known_word_offset)
        # skipmatch_count = 0
        known_word = None
        if include_variants is None:
            include_variants = self.include_variants
        if known_word_start_offset is None:
            known_word_start_offset = {}
        # print(known_word_start_offset)
        skip_matches = SkipMatches(self.ngram_size, self.skip_size)
        text_string = text['text'].lower() if self.ignorecase else text['text']
        for skipgram in text2skipgrams(text_string, self.ngram_size, self.skip_size):
            # print(skipgram.start_offset, skipgram.string)
            # print("skipgram:", skipgram.string)
            if skipgram.start_offset in known_word_start_offset:
                known_word = known_word_start_offset[skipgram.start_offset]
                # print("known word start_offset reached:", known_word)
            if known_word and skipgram.start_offset == known_word["end"]:
                # print("end of known word start_offset reached:", known_word)
                known_word = None
            for phrase in self.skipgram_index[skipgram.string]:
                if phrase.max_start_offset > 0 and phrase.max_start_end < skipgram.start_offset + \
                        skipgram.length + self.max_length_variance:
                    # print(skipgram.offset, phrase.max_start_offset, phrase.max_start_end, phrase.phrase_string)
                    # print(f"skipping phrase {phrase.phrase_string} at offset", skipgram.offset)
                    continue
                if phrase.max_end_offset > 0 and phrase.max_end_end < skipgram.end_offset + \
                        skipgram.length + self.max_length_variance:
                    # print(skipgram.offset, phrase.max_end_offset, phrase.max_end_end, phrase.phrase_string)
                    # print(f"skipping phrase {phrase.phrase_string} at offset", skipgram.offset)
                    continue
                if known_word:
                    #if phrase.phrase_string not in self.phrase_model.word_in_phrase[known_word["word"]]:
                    #    print("skipping phrase because doesn't match known word:", phrase.phrase_string)
                    #    continue
                    if phrase.phrase_string in known_word["match_phrases"]:
                        # print("skipping phrase because found as exact match:", phrase.phrase_string)
                        continue
                # print("\tphrase has skip:", phrase.phrase_string)
                # skipmatch_count += 1
                # print("adding skipmatch", skipmatch_count)
                skip_matches.add_skip_match(skipgram, phrase)
            if include_variants:
                for phrase in self.variant_skipgram_index[skipgram.string]:
                    if known_word:
                        if phrase.phrase_string not in self.phrase_model.word_in_phrase[known_word["word"]]:
                            # print("skipping phrase because doesn't match known word:", phrase.phrase_string)
                            continue
                        if phrase.phrase_string in known_word["match_phrases"]:
                            # print("skipping phrase because found as exact match:", phrase.phrase_string)
                            continue
                    skip_matches.add_skip_match(skipgram, phrase)
        # print("final skipmatch count:", skipmatch_count)
        return skip_matches

    @staticmethod
    def filter_matches_by_offset_threshold(matches: List[PhraseMatch], debug: int = 0):
        filtered_matches = []
        for match in matches:
            if debug > 1:
                print('searcher.filter_matches_by_offset_threshold - match:\n\t', match.phrase.phrase_string,
                      match.phrase.max_start_offset, match.offset)
            if match.phrase.max_start_offset is None or match.phrase.max_start_offset == -1:
                if debug > 1:
                    print('no max start')
                filtered_matches.append(match)
            elif match.phrase.max_start_offset >= match.offset:
                if debug > 1:
                    print('lower than max start')
                filtered_matches.append(match)
            else:
                if debug > 1:
                    print('skipping')
                continue
        return filtered_matches
