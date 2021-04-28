from typing import Dict, List, Set, Union
import copy
import string
import re
from collections import defaultdict

from fuzzy_search.fuzzy_phrase_model import PhraseModel
from fuzzy_search.fuzzy_match import PhraseMatch, Candidate, adjust_match_offsets
from fuzzy_search.fuzzy_phrase import Phrase
from fuzzy_search.fuzzy_string import text2skipgrams, SkipGram, score_levenshtein_similarity_ratio


default_config = {
    # these thresholds work when there are quite a few OCR errors
    # use higher thresholds for higher quality OCR to avoid
    # spurious matches.
    "char_match_threshold": 0.6,
    "ngram_threshold": 0.5,
    "levenshtein_threshold": 0.6,
    "skipgram_threshold": 0.2,
    # Is upper/lowercase a meaningful signal?
    "ignorecase": False,
    # should matches follow word boundaries?
    "use_word_boundaries": False,
    # for phrases that have variant phrasings
    "include_variants": False,
    # avoid matching with similar but different phrases
    "filter_distractors": False,
    # matching string can be lower/shorter than prhase
    "max_length_variance": 1,
    # higher ngram size allows fewer character differences
    "ngram_size": 2,
    # fewer skips is much faster but less exhaustive
    "skip_size": 2,
    # first check for exact matches to speed up fuzzy search
    "skip_exact_matching": False,
    # allow matches of partially overlapping phrase
    "allow_overlapping_matches": False,
    # the set of symbols to use as punctuation (for word boundaries)
    "punctuation": string.punctuation
}


class SkipMatches:

    def __init__(self, ngram_size: int, skip_size: int):
        self.ngram_size = ngram_size
        self.skip_size = skip_size
        self.skip_length = ngram_size + skip_size
        self.match_set: Dict[Phrase, set] = defaultdict(set)
        self.match_offsets = defaultdict(list)
        self.match_skipgrams: Dict[Phrase, List[SkipGram]] = defaultdict(list)
        self.phrases: Set[Phrase] = set()

    def add_skip_match(self, skipgram: SkipGram, phrase: Phrase) -> None:
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
        self.phrases.add(phrase)


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
    return [phrase for phrase in skip_matches.phrases if get_skipset_overlap(phrase, skip_matches) >= skip_threshold]


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
                                    skipgram_threshold: float, max_length_variance: int = 1) -> List[Candidate]:
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
    :return: a list of candidate matches
    :rtype: List[Candidate]
    """
    candidates: List[Candidate] = []
    candidate = Candidate(phrase, max_length_variance=max_length_variance)
    last_index = len(skip_matches.match_offsets[phrase]) - 1
    # print(f"finding candidates for phrase ({len(phrase.phrase_string)}):", phrase.phrase_string)
    for ci, curr_offset in enumerate(skip_matches.match_offsets[phrase]):
        next_offset = None if ci == last_index else skip_matches.match_offsets[phrase][ci + 1]
        # print(ci, 'curr offset:', curr_offset, '\tskip:',
        #       skip_matches.match_skipgrams[phrase][ci].string, '\tnext offset:', next_offset)
        # add current skipgram to the candidate
        candidate.add_skip_match(skip_matches.match_skipgrams[phrase][ci])
        # if abs(candidate.skip_match_length() - len(candidate.phrase.phrase_string)) < max_length_variance:
        #     skip = skip_matches.match_skipgrams[phrase][ci]
        #     print(ci, curr_offset, "adding skip match:", skip.string, skip.offset, skip.length)
        #     print("candidate skips:", [skip.string for skip in candidate.skipgram_list],
        #           candidate.skip_match_length())
        #     print(candidate.get_skip_set_overlap(), candidate.get_match_string(text))
        # check if the current candidate is a potential match for the phrase
        if candidate.is_match(skipgram_threshold):
            candidate.match_string = candidate.get_match_string(text)
            # print("meets threshold:", candidate.match_string)
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
            # print('curr_offset:', curr_offset, '\tnext_offset:', next_offset)
            # print('starting a new candidate')
            candidate = Candidate(phrase)
    # end of skipgrams reached, check if remaining candidate is a match
    # print('checking if final candidate is match')
    if candidate.is_match(skipgram_threshold):
        if len(candidates) == 0 or not candidate.same_candidate(candidates[-1]):
            candidate.match_string = candidate.get_match_string(text)
            candidates.append(copy.deepcopy(candidate))
        if candidate.shift_start_skip():
            # candidate string is longer than phrase string check if shifting the start creates
            # a better candidate and if so, add that as well
            candidate.match_string = candidate.get_match_string(text)
            candidates.append(copy.deepcopy(candidate))
    return candidates


def get_skipmatch_candidates(text: Dict[str, any], skip_matches: SkipMatches,
                             skipgram_threshold: float, phrase_model: PhraseModel,
                             max_length_variance: int = 1) -> List[Candidate]:
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
    :return: a list of candidate matches
    :rtype: List[Candidate]
    """
    phrase_candidates = defaultdict(list)
    candidates: List[Candidate] = []
    for phrase in skip_matches.phrases:
        # print("get_skipmatch_candidates - phrase:", phrase.phrase_string)
        if get_skipset_overlap(phrase, skip_matches) < skipgram_threshold:
            continue
        if phrase.phrase_string in phrase_model.is_variant_of:
            match_phrase = phrase_model.is_variant_of[phrase.phrase_string]
        else:
            match_phrase = phrase.phrase_string
        phrase_candidates[match_phrase] += get_skipmatch_phrase_candidates(text, phrase, skip_matches,
                                                                           skipgram_threshold,
                                                                           max_length_variance=max_length_variance)
    for phrase_string in phrase_candidates:
        # print("phrase_candidates:", len(phrase_candidates[phrase_string]))
        filtered_candidates = filter_overlapping_phrase_candidates(phrase_candidates[phrase_string])
        # print(phrase_candidates)
        # print("filtered_candidates:", len(filtered_candidates))
        # for candidate in filtered_candidates:
        #     print(candidate.match_string, candidate.match_start_offset, candidate.match_end_offset)
        candidates += filtered_candidates
    return candidates


def get_text_dict(text: Union[str, dict], ignorecase: bool = False) -> dict:
    """Check that text is in a dictionary with an id property, so that passing a long text
    goes by reference instead of copying the long text string.

    :param text: a text string or text dictionary
    :type text: Union[str, dict]
    :param ignorecase: boolean flag for whether to ignore case
    :type ignorecase: bool
    :return: a text dictionary with an id property
    :rtype: dict
    """
    if isinstance(text, str):
        text = {"text": text, "id": None}
    if ignorecase:
        text["text"] = text["text"].lower()
    if "id" not in text:
        text["id"] = None
    return text


def candidates_to_matches(candidates: List[Candidate], text: dict, phrase_model: PhraseModel) -> List[PhraseMatch]:
    matches: List[PhraseMatch] = []
    for candidate in candidates:
        if candidate.phrase.phrase_string in phrase_model.is_variant_of:
            match_phrase_string = phrase_model.is_variant_of[candidate.phrase.phrase_string]
            match_phrase = phrase_model.phrase_index[match_phrase_string]
        else:
            match_phrase = candidate.phrase
        match = PhraseMatch(match_phrase, candidate.phrase,
                            candidate.match_string, candidate.match_start_offset, text["id"])
        match.add_scores(skipgram_overlap=candidate.get_skip_count_overlap())
        matches.append(match)
    return matches


def filter_matches_by_overlap(filtered_matches: List[PhraseMatch]) -> List[PhraseMatch]:
    sorted_matches = sorted(filtered_matches, key=lambda x: (x.offset, len(x.string)))
    filtered_matches = []
    overlapping = defaultdict(list)
    for match in sorted_matches:
        overlapping[(match.offset, len(match.string))].append(match)
    for offset_length in overlapping:
        if len(overlapping[offset_length]) == 1:
            filtered_matches.extend(overlapping[offset_length])
        else:
            best = max(overlapping[offset_length], key=lambda item: item.levenshtein_similarity)
            filtered_matches.append(best)
    return filtered_matches


class FuzzyPhraseSearcher(object):

    def __init__(self, config: Union[None, Dict[str, Union[str, int, float]]] = None):
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
        if config:
            self.configure(config)

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
        if "allow_overlapping_matches" in config:
            self.allow_overlapping_matches = config["allow_overlapping_matches"]
        if "punctuation" in config:
            self.punctuation = config["punctuation"]
        if "debug" in config:
            self.debug = config["debug"]

    def index_phrase_model(self, phrase_model: Union[List[Dict[str, Union[str, int, float, list]]], PhraseModel]):
        """Add a phrase model to search for phrases in texts.

        :param phrase_model: a phrase model, either as dictionary or as PhraseModel object
        :type phrase_model: Union[List[Dict[str, Union[str, int, float, list]]], PhraseModel]
        """
        if isinstance(phrase_model, list):
            phrase_model = PhraseModel(model=phrase_model, config=self.config)
        self.phrase_model = phrase_model
        self.index_phrases(list(phrase_model.phrase_index.values()))
        self.index_variants(list(phrase_model.variant_index.values()))
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
                for skipgram in phrase.skipgrams_lower:
                    self.skipgram_index[skipgram.string].add(phrase)
                for skipgram_string in phrase.early_skipgram_index_lower:
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
                              known_word_offset: Dict[int, Dict[str, any]] = None) -> SkipMatches:
        """Find all skipgram matches between text and phrases.

        :param text: the text object to match with phrases
        :type text: Dict[str, Union[str, int, float, list]]
        :param include_variants: boolean flag for whether to include phrase variants for finding matches
        :type include_variants: bool
        :param known_word_offset: a dictionary of known words and their text offsets based on exact matches
        :type known_word_offset: Dict[int, Dict[str, any]]
        :return: a SkipMatches object contain all skipgram matches
        :rtype: SkipMatches
        """
        # skipmatch_count = 0
        known_word = None
        if include_variants is None:
            include_variants = self.include_variants
        if known_word_offset is None:
            known_word_offset = {}
        # print(known_word_offset)
        skip_matches = SkipMatches(self.ngram_size, self.skip_size)
        for skipgram in text2skipgrams(text["text"], self.ngram_size, self.skip_size):
            # print(skipgram.offset, skipgram.string)
            # print("skipgram:", skipgram.string)
            if skipgram.offset in known_word_offset:
                known_word = known_word_offset[skipgram.offset]
                # print("known word offset reached:", known_word)
            if known_word and skipgram.offset == known_word["end"]:
                # print("end of known word offset reached:", known_word)
                known_word = None
            """
            if known_word:
                for phrase_string in self.phrase_model.word_in_phrase[known_word["word"]]:
                    if phrase_string in known_word["match_phrases"]:
                        # this phrase was already found through exact match
                        continue
                    if phrase_string in self.phrase_model.phrase_index:
                        phrase = self.phrase_model.phrase_index[phrase_string]
                    elif include_variants and phrase_string in self.phrase_model.variant_index:
                        phrase = self.phrase_model.variant_index[phrase_string]
                    else:
                        # the phrase string either is a variant which is not include, or a distractor
                        continue
                    if skipgram.string in phrase.skipgrams:
                        skip_matches.add_skip_match(skipgram, phrase)
            else:
                for phrase in self.skipgram_index[skipgram.string]:
                    skip_matches.add_skip_match(skipgram, phrase)
                if include_variants:
                    for phrase in self.variant_skipgram_index[skipgram.string]:
                        skip_matches.add_skip_match(skipgram, phrase)
                """
            for phrase in self.skipgram_index[skipgram.string]:
                if phrase.max_offset > 0 and phrase.max_end < skipgram.offset + \
                        skipgram.length + self.max_length_variance:
                    # print(skipgram.offset, phrase.max_offset, phrase.max_end, phrase.phrase_string)
                    # print(f"skipping phrase {phrase.phrase_string} at offset", skipgram.offset)
                    continue
                if known_word:
                    if phrase.phrase_string not in self.phrase_model.word_in_phrase[known_word["word"]]:
                        # print("skipping phrase because doesn't match known word:", phrase.phrase_string)
                        continue
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

    def find_candidates(self, text: dict, use_word_boundaries: bool,
                        include_variants: Union[None, bool] = None,
                        known_word_offset: Dict[int, Dict[str, any]] = None) -> List[Candidate]:
        """Find candidate fuzzy matches for a given text.

        :param text: the text object to match with phrases
        :type text: dict
        :param use_word_boundaries: use word boundaries in determining match boundaries
        :type use_word_boundaries: bool
        :param include_variants: boolean flag for whether to include phrase variants for finding matches
        :type include_variants: bool
        :param known_word_offset: a dictionary of known words and their text offsets based on exact matches
        :type known_word_offset: Dict[int, Dict[str, any]]
        :return: a list of candidate matches
        :rtype: List[Candidate]
        """
        skip_matches = self.find_skipgram_matches(text, include_variants=include_variants,
                                                  known_word_offset=known_word_offset)
        candidates = get_skipmatch_candidates(text, skip_matches, self.skipgram_threshold, self.phrase_model,
                                              max_length_variance=self.max_length_variance)
        filtered = []
        use_word_boundaries = use_word_boundaries if use_word_boundaries is not None else self.use_word_boundaries
        for candidate in candidates:
            # print(candidate)
            if use_word_boundaries:
                adjusted_match = adjust_match_offsets(candidate.phrase.phrase_string, candidate.match_string,
                                                      text, candidate.match_start_offset, candidate.match_end_offset,
                                                      self.punctuation)
                # print("adjusted_match:", adjusted_match)
                if not adjusted_match:
                    continue
                candidate.match_start_offset = adjusted_match["match_start_offset"]
                candidate.match_end_offset = adjusted_match["match_end_offset"]
                candidate.match_string = adjusted_match["match_string"]
                # print("new match string:", candidate.match_string)
            filtered.append(candidate)
        return filtered

    def filter_matches_by_distractors(self, matches: List[PhraseMatch]) -> List[PhraseMatch]:
        filtered: List[PhraseMatch] = []
        for match in matches:
            if match.phrase.phrase_string in self.phrase_model.has_distractors:
                for distractor in self.phrase_model.has_distractors[match.phrase.phrase_string]:
                    score = score_levenshtein_similarity_ratio(match.string, distractor)
                    if score > match.levenshtein_similarity:
                        break
                else:
                    filtered.append(match)
            else:
                filtered.append(match)
        return filtered

    def filter_matches_by_threshold(self, matches: List[PhraseMatch]) -> List[PhraseMatch]:
        filtered: List[PhraseMatch] = []
        for match in matches:
            if match.character_overlap < self.char_match_threshold:
                continue
            if match.ngram_overlap < self.ngram_threshold:
                continue
            if match.levenshtein_similarity < self.levenshtein_threshold:
                continue
            filtered.append(match)
        return filtered

    def find_matches(self, text: Union[str, Dict[str, str]],
                     use_word_boundaries: Union[None, bool] = None,
                     allow_overlapping_matches: Union[None, bool] = None,
                     include_variants: Union[None, bool] = None,
                     filter_distractors: Union[None, bool] = None,
                     skip_exact_matching: bool = None) -> List[PhraseMatch]:
        """Find all fuzzy matching phrases for a given text. By default, a first pass of exact matching is conducted
        to find exact occurrences of phrases. This is to speed up the fuzzy matching pass

        :param text: the text (string or dictionary with 'text' property) to find fuzzy matching phrases in.
        :type text: Union[str, Dict[str, str]]
        :param use_word_boundaries: use word boundaries in determining match boundaries
        :type use_word_boundaries: Union[None, bool]
        :param allow_overlapping_matches: boolean flag for whether to allow matches to overlap in their text ranges
        :type allow_overlapping_matches: Union[None, bool]
        :param include_variants: boolean flag for whether to include phrase variants for finding matches
        :type include_variants: Union[None, bool]
        :param filter_distractors: boolean flag for whether to remove phrase matches that better match distractors
        :type filter_distractors: Union[None, bool]
        :param skip_exact_matching: boolean flag whether to skip the exact matching step
        :type skip_exact_matching: Union[None, bool]
        :return: a list of phrases matches
        :rtype: PhraseMatch
        """
        if self.phrase_model is None:
            raise ValueError("No phrase model indexed")
        text = get_text_dict(text, ignorecase=self.ignorecase)
        if use_word_boundaries is None:
            use_word_boundaries = self.use_word_boundaries
        if skip_exact_matching is None:
            skip_exact_matching = self.skip_exact_matching
        if not skip_exact_matching:
            # print("running exact matching")
            exact_matches = self.find_exact_matches(text, use_word_boundaries=use_word_boundaries,
                                                    include_variants=include_variants)
            known_word_offset = index_known_word_offsets(exact_matches)
        else:
            # print("skipping exact matching")
            exact_matches = []
            known_word_offset = {}
        # print('number of exact matches:', len(exact_matches))
        candidates = self.find_candidates(text, use_word_boundaries=use_word_boundaries,
                                          include_variants=include_variants, known_word_offset=known_word_offset)
        # print(candidates)
        matches = candidates_to_matches(candidates, text, self.phrase_model)
        # print(matches)
        filtered_matches = self.filter_matches_by_threshold(matches)
        if filter_distractors is None:
            filter_distractors = self.filter_distractors
        if filter_distractors:
            filtered_matches = self.filter_matches_by_distractors(filtered_matches)
        if allow_overlapping_matches is None:
            allow_overlapping_matches = self.allow_overlapping_matches
        if not allow_overlapping_matches:
            filtered_matches = filter_matches_by_overlap(filtered_matches)
        # print(exact_matches)
        # print(filtered_matches)
        selected_matches = filtered_matches + exact_matches
        return sorted(selected_matches, key=lambda x: x.offset)

    def find_exact_matches(self, text: Union[str, Dict[str, str]],
                           use_word_boundaries: Union[None, bool] = None,
                           include_variants: Union[None, bool] = None) -> List[PhraseMatch]:
        """Find all fuzzy matching phrases for a given text.

        :param text: the text (string or dictionary with 'text' property) to find fuzzy matching phrases in.
        :type text: Union[str, Dict[str, str]]
        :param use_word_boundaries: use word boundaries in determining match boundaries
        :type use_word_boundaries: Union[None, bool]
        :param include_variants: boolean flag for whether to include phrase variants for finding matches
        :type include_variants: Union[None, bool]
        :return: a list of phrases matches
        :rtype: PhraseMatch
        """
        exact_matches: List[PhraseMatch] = []
        text = get_text_dict(text, ignorecase=self.ignorecase)
        if use_word_boundaries is None:
            use_word_boundaries = self.use_word_boundaries
        if include_variants is None:
            include_variants = self.include_variants
        for exact_match in search_exact_phrases(self.phrase_model, text, use_word_boundaries=use_word_boundaries,
                                                include_variants=include_variants):
            exact_matches.append(exact_match)
        return exact_matches


def index_known_word_offsets(exact_matches: List[PhraseMatch]) -> Dict[int, Dict[str, any]]:
    exact_match_offset: Dict[int, Set[PhraseMatch]] = defaultdict(set)
    known_word_offset: Dict[int, Dict[str, any]] = defaultdict(dict)
    for exact_match in exact_matches:
        match_words = re.split(r"\W+", exact_match.string)
        # print("exact match:", exact_match)
        text_offset = exact_match.offset
        match_offset = 0
        # print("text_offset:", text_offset, "\tmatch_offset:", match_offset)
        for match_word in match_words:
            start = text_offset + match_offset + exact_match.string[match_offset:].index(match_word)
            # print("match_word:", match_word, "\tstart:", start)
            end = start + len(match_word)
            if start not in known_word_offset:
                known_word = {
                    "word": match_word,
                    "start": start,
                    "end": end,
                    "match_phrases": {exact_match.string}
                }
                # print(known_word)
                known_word_offset[start] = known_word
            known_word_offset[start]["match_phrases"].add(exact_match.string)
            match_word_offset = match_offset + exact_match.string[match_offset:].index(match_word)
            # print("text_offset:", text_offset, "\tmatch_offset:", match_offset)
        exact_match_offset[exact_match.offset].add(exact_match)
    return known_word_offset


def search_exact_phrases(phrase_model: PhraseModel, text: Dict[str, str],
                         ignorecase: bool = False, use_word_boundaries: bool = True,
                         include_variants: bool = False):
    if use_word_boundaries:
        # print('searching with word boundaries')
        return search_exact_phrases_with_word_boundaries(phrase_model, text, ignorecase=ignorecase,
                                                         include_variants=include_variants)
    else:
        return search_exact_phrases_without_word_boundaries(phrase_model, text, ignorecase=ignorecase,
                                                            include_variants=include_variants)


def add_exact_match_score(match: PhraseMatch) -> PhraseMatch:
    match.character_overlap = 1.0
    match.ngram_overlap = 1.0
    match.levenshtein_similarity = 1.0
    return match


def search_exact_phrases_with_word_boundaries(phrase_model: PhraseModel, text: Dict[str, str],
                                              ignorecase: bool = False, include_variants: bool = False):
    for word in re.finditer(r"\w+", text["text"]):
        if word.group(0) not in phrase_model.word_in_phrase:
            continue
        # print("\tword:", word)
        for phrase_string in phrase_model.first_word_in_phrase[word.group(0)]:
            phrase_word_offset = phrase_model.first_word_in_phrase[word.group(0)][phrase_string]
            phrase_start = word.start() - phrase_word_offset
            phrase_end = phrase_start + len(phrase_string)
            # print(phrase_start, phrase_end, phrase_string)
            if text["text"][phrase_start:phrase_end] == phrase_string:
                if phrase_start > 0 and re.match(r'\w', text["text"][phrase_start - 1]):
                    continue
                if phrase_end < len(text['text']) - 1 and re.match(r'\w', text['text'][phrase_end]):
                    continue
                if "phrase" in phrase_model.phrase_type[phrase_string]:
                    phrase = phrase_model.phrase_index[phrase_string]
                    match = PhraseMatch(phrase, phrase, phrase_string, phrase_start, text_id=text["id"])
                    yield add_exact_match_score(match)
                    # print("the matching phrase:", phrase)
                elif "variant" in phrase_model.phrase_type[phrase_string] and include_variants:
                    variant_phrase = phrase_model.variant_index[phrase_string]
                    main_phrase_string = phrase_model.is_variant_of[phrase_string]
                    main_phrase = phrase_model.phrase_index[main_phrase_string]
                    match = PhraseMatch(main_phrase, variant_phrase, phrase_string, phrase_start, text_id=text["id"])
                    yield add_exact_match_score(match)


def search_exact_phrases_without_word_boundaries(phrase_model: PhraseModel, text: Dict[str, str],
                                                 ignorecase: bool = False,
                                                 include_variants: bool = False):
    for phrase_string in phrase_model.phrase_index:
        phrase = phrase_model.phrase_index[phrase_string]
        for match in re.finditer(phrase.exact_string, text["text"]):
            phrase = phrase_model.phrase_index[phrase_string]
            match = PhraseMatch(phrase, phrase, phrase_string, match.start(), text_id=text["id"])
            yield add_exact_match_score(match)
    if include_variants:
        for phrase_string in phrase_model.variant_index:
            variant_phrase = phrase_model.variant_index[phrase_string]
            for match in re.finditer(variant_phrase.exact_string, text["text"]):
                variant_phrase = phrase_model.variant_index[phrase_string]
                main_phrase_string = phrase_model.is_variant_of[phrase_string]
                main_phrase = phrase_model.phrase_index[main_phrase_string]
                match = PhraseMatch(main_phrase, variant_phrase, phrase_string, match.start(), text_id=text["id"])
                yield add_exact_match_score(match)


def search_exact(phrase: Phrase, text: Dict[str, str], ignorecase: bool = False, use_word_boundaries: bool = True):
    search_string = phrase.extact_word_boundary_string if use_word_boundaries else phrase.exact_string
    if ignorecase:
        return re.finditer(search_string, text["text"], flags=re.IGNORECASE)
    else:
        return re.finditer(search_string, text["text"])


def get_known_word_offsets(match_ranges: List[Dict[str, any]], text_doc: Dict[str, str]) -> Dict[int, dict]:
    known_word_offset = {}
    offset = match_ranges[0]["s"]
    for match_range in match_ranges:
        print(match_range)
        print("offset:", offset)
        match_text = text_doc["text"][match_range["s"]:match_range["e"]]
        match_words = re.split(r"\W+", match_text)
        for match_word in match_words:
            start = offset + text_doc["text"][offset:].index(match_word)
            end = start + len(match_word)
            known_word = {
                "match_word": match_word,
                "start": start,
                "end": end,
                "match_phrases": set([phrase for phrase in match_range["phrases"]])
            }
            match_word_offset = offset + text_doc["text"][offset:].index(match_word)
            offset = match_word_offset + len(match_word)
            print(known_word)
            known_word_offset[known_word["start"]] = known_word
    return known_word_offset


def get_exact_match_ranges(exact_matches: List[PhraseMatch]) -> List[dict]:
    sorted_matches = sorted(exact_matches, key=lambda m: m.offset)
    match_ranges = []
    if len(exact_matches) == 0:
        return []
    first_match = exact_matches[0]
    match_range = {"s": first_match.offset, "e": first_match.end, "phrases": {first_match.phrase.phrase_string}}
    for mi, match in enumerate(sorted_matches[1:]):
        if match.offset > match_range["e"]:
            match_ranges.append(match_range)
            match_range = {"s": match.offset, "e": match.end, "phrases": set()}
        match_range["e"] = max(match_range["e"], match.end)
        match_range["phrases"].add(match.phrase.phrase_string)
        if match.phrase.phrase_string != match.variant.phrase_string:
            match_range["phrases"].add(match.variant.phrase_string)
    match_ranges.append(match_range)
    return match_ranges
