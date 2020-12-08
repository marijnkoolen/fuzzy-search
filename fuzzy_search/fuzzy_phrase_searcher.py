from typing import Dict, List, Set, Union
import copy
from collections import defaultdict

from fuzzy_search.fuzzy_phrase_model import PhraseModel
from fuzzy_search.fuzzy_match import Match, Candidate, adjust_match_offsets
from fuzzy_search.fuzzy_phrase import Phrase
from fuzzy_search.fuzzy_string import text2skipgrams, SkipGram, score_levenshtein_similarity_ratio


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
    prev_candidate = phrase_candidates[0]
    prev_score = score_levenshtein_similarity_ratio(prev_candidate.phrase.phrase_string, prev_candidate.match_string)
    for ci, curr_candidate in enumerate(phrase_candidates[1:]):
        # print("prev_candidate:", prev_candidate)
        # print("curr_candidate:", curr_candidate)
        if curr_candidate.match_end_offset > prev_candidate.match_start_offset:
            if curr_candidate.match_start_offset < prev_candidate.match_end_offset:
                # print("CONSIDERING CURR")
                # this candidate overlaps with the previous one, pick the best
                # print(prev_candidate.get_skip_count_overlap(), curr_candidate.get_skip_count_overlap())
                curr_score = score_levenshtein_similarity_ratio(curr_candidate.phrase.phrase_string,
                                                                curr_candidate.match_string)
                # if curr_candidate.get_skip_count_overlap() > prev_candidate.get_skip_count_overlap():
                if curr_score > prev_score:
                    # print("SELECTING CURR")
                    # this candidate is better, so skip the previous candidate
                    prev_candidate = curr_candidate
                    prev_score = curr_score
            else:
                # the previous candidate does not overlap with the current, so add it to filtered
                # print("APPENDING")
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
    # print("generating candidate:", candidate.match_string)
    last_index = len(skip_matches.match_offsets[phrase]) - 1
    for ci, curr_offset in enumerate(skip_matches.match_offsets[phrase]):
        next_offset = None if ci == last_index else skip_matches.match_offsets[phrase][ci + 1]
        # add current skipgram to the candidate
        skip = skip_matches.match_skipgrams[phrase][ci]
        skip_list = [skip.string for skip in candidate.skipgram_list]
        # print("\t", skip.string, skip.offset, candidate.get_match_string(text), skip_list, candidate.get_skip_set_overlap())
        candidate.add_skip_match(skip_matches.match_skipgrams[phrase][ci])
        if candidate.is_match(skipgram_threshold):
            candidate.match_string = candidate.get_match_string(text)
            # if this candidate has enough skipgram overlap, yield it as a candidate match
            if len(candidates) == 0 or not candidate.same_candidate(candidates[-1]):
                candidates.append(copy.deepcopy(candidate))
                # print(ci, "appending:", candidate.match_string)
            if candidate.skip_match_length() > len(phrase.phrase_string):
                if candidate.shift_start_skip():
                    candidate.match_string = candidate.get_match_string(text)
                    candidates.append(copy.deepcopy(candidate))
                    # print(ci, "appending better start:", candidate.match_string)
        if next_offset and next_offset - curr_offset > skip_matches.ngram_size + skip_matches.skip_size:
            # if the gap between the current skipgram and the next is larger than an entire skipgram
            # the next skipgram does not belong to this candidate
            # start a new candidate for the next skipgram
            # print("GAP")
            candidate = Candidate(phrase)
            # print("generating candidate:", candidate.match_string)
    # end of skipgrams reached, check if remaining candidate is a match
    if candidate.is_match(skipgram_threshold):
        if len(candidates) == 0 or not candidate.same_candidate(candidates[-1]):
            candidate.match_string = candidate.get_match_string(text)
            candidates.append(copy.deepcopy(candidate))
    return candidates


def get_skipmatch_candidates(text: Dict[str, any], skip_matches: SkipMatches,
                             skipgram_threshold: float, max_length_variance: int = 1) -> List[Candidate]:
    """Find all candidate matches for the phrases in a SkipMatches object.

    :param text: the text object to match with phrases
    :type text: Dict[str, any]
    :param skip_matches: a SkipMatches object with matches between a text and a list of phrases
    :type skip_matches: SkipMatches
    :param skipgram_threshold: a threshold for how many skipgrams should match between a phrase and a candidate
    :type skipgram_threshold: float
    :param max_length_variance: the maximum difference in length between candidate and phrase
    :type max_length_variance: int
    :return: a list of candidate matches
    :rtype: List[Candidate]
    """
    candidates: List[Candidate] = []
    for phrase in skip_matches.phrases:
        # print("get_skipmatch_candidates - phrase:", phrase.phrase_string)
        if get_skipset_overlap(phrase, skip_matches) < skipgram_threshold:
            continue
        phrase_candidates = get_skipmatch_phrase_candidates(text, phrase, skip_matches, skipgram_threshold,
                                                            max_length_variance=max_length_variance)
        # print(phrase_candidates)
        phrase_candidates = filter_overlapping_phrase_candidates(phrase_candidates)
        # print(phrase_candidates)
        candidates += phrase_candidates
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


def candidates_to_matches(candidates: List[Candidate], text: dict, phrase_model: PhraseModel) -> List[Match]:
    matches: List[Match] = []
    for candidate in candidates:
        if candidate.phrase.phrase_string in phrase_model.is_variant_of:
            match_phrase_string = phrase_model.is_variant_of[candidate.phrase.phrase_string]
            match_phrase = phrase_model.phrase_index[match_phrase_string]
        else:
            match_phrase = candidate.phrase
        match = Match(match_phrase, candidate.phrase,
                      candidate.match_string, candidate.match_start_offset, text["id"])
        match.add_scores(skipgram_overlap=candidate.get_skip_count_overlap())
        matches.append(match)
    return matches


class FuzzyPhraseSearcher(object):

    def __init__(self, config: Union[None, Dict[str, Union[str, int, float]]] = None):
        # default configuration
        self.char_match_threshold = 0.5
        self.ngram_threshold = 0.5
        self.skipgram_threshold = 0.3
        self.levenshtein_threshold = 0.5
        self.perform_strip_suffix = True
        self.max_length_variance = 1
        self.use_word_boundaries = True
        self.ignorecase = False
        self.track_candidates = False
        self.use_confuse = False
        self.tracking_level = 4
        self.known_candidates = defaultdict(dict)
        self.distractor_terms = defaultdict(list)
        self.ngram_size = 2
        self.skipgram_index = defaultdict(list)
        self.early_skipgram_index = defaultdict(list)
        self.late_skipgram_index = defaultdict(list)
        self.skip_size = 2
        self.variant_map = defaultdict(dict)
        self.has_variant = defaultdict(dict)
        self.variant_skipgram_index = defaultdict(list)
        self.variant_early_skipgram_index = defaultdict(list)
        self.variant_late_skipgram_index = defaultdict(list)
        self.include_variants = False
        self.filter_distractors = False
        self.phrases: Set[Phrase] = set()
        self.variants: Set[Phrase] = set()
        self.phrase_model: Union[None, PhraseModel] = None
        self.debug = False
        # non-default configuration
        if config:
            self.config = config
            self.configure(config)

    def configure(self, config: Dict[str, Union[str, int, float]]) -> None:
        """Configure the fuzzy searcher with a given config object.

        :param config: a config dictionary
        :type config: Dict[str, Union[str, int, float]]
        """
        self.config = config
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
        if "track_candidates" in config:
            self.track_candidates = config["track_candidates"]
        if "use_confuse" in config:
            self.use_confuse = config["use_confuse"]
        if "ngram_size" in config:
            self.ngram_size = config["ngram_size"]
        if "skip_size" in config:
            self.skip_size = config["skip_size"]
        if "include_variants" in config:
            self.include_variants = config["include_variants"]
        if "debug" in config:
            self.debug = config["debug"]

    def set_strip_suffix(self, strip_suffix: bool) -> None:
        """Set boolean for whether match strings should be stripped to word boundaries.

        :param strip_suffix: boolean for toggling match string suffix stripping
        :type strip_suffix: bool
        :return: None
        :rtype: None
        """
        self.perform_strip_suffix = strip_suffix

    def index_phrase_model(self, phrase_model: Union[List[Dict[str, Union[str, int, float, list]]], PhraseModel]):
        """Add a phrase model to search for phrases in texts.

        :param phrase_model: a phrase model, either as dictionary or as PhraseModel object
        :type phrase_model: Union[List[Dict[str, Union[str, int, float, list]]], PhraseModel]
        """
        if isinstance(phrase_model, list):
            phrase_model = PhraseModel(model=phrase_model)
        self.phrase_model = phrase_model
        self.index_phrases(list(phrase_model.phrase_index.values()))
        self.index_variants(list(phrase_model.variant_index.values()))

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
                    self.skipgram_index[skipgram.string].append(phrase)
                for skipgram_string in phrase.early_skipgram_index_lower:
                    self.early_skipgram_index[skipgram_string].append(phrase)
                for skipgram_string in phrase.late_skipgram_index_lower:
                    self.late_skipgram_index[skipgram_string].append(phrase)
            else:
                for skipgram in phrase.skipgrams:
                    self.skipgram_index[skipgram.string].append(phrase)
                for skipgram_string in phrase.early_skipgram_index:
                    self.early_skipgram_index[skipgram_string].append(phrase)
                for skipgram_string in phrase.late_skipgram_index:
                    self.late_skipgram_index[skipgram_string].append(phrase)
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
                    self.variant_skipgram_index[skipgram.string].append(variant)
                for skipgram_string in variant.early_skipgram_index_lower:
                    self.variant_early_skipgram_index[skipgram_string].append(variant)
                for skipgram_string in variant.late_skipgram_index_lower:
                    self.variant_late_skipgram_index[skipgram_string].append(variant)
            else:
                for skipgram in variant.skipgrams:
                    self.variant_skipgram_index[skipgram.string].append(variant)
                for skipgram_string in variant.early_skipgram_index:
                    self.variant_early_skipgram_index[skipgram_string].append(variant)
                for skipgram_string in variant.late_skipgram_index:
                    self.variant_late_skipgram_index[skipgram_string].append(variant)

    def find_skipgram_matches(self, text: Dict[str, Union[str, int, float, list]],
                              include_variants: Union[None, bool] = None) -> SkipMatches:
        """Find all skipgram matches between text and phrases.

        :param text: the text object to match with phrases
        :type text: Dict[str, Union[str, int, float, list]]
        :param include_variants: boolean flag for whether to include phrase variants for finding matches
        :type include_variants: bool
        :return: a SkipMatches object contain all skipgram matches
        :rtype: SkipMatches
        """
        if include_variants is None:
            include_variants = self.include_variants
        skip_matches = SkipMatches(self.ngram_size, self.skip_size)
        for skipgram in text2skipgrams(text["text"], self.ngram_size, self.skip_size):
            for phrase in self.skipgram_index[skipgram.string]:
                skip_matches.add_skip_match(skipgram, phrase)
            if include_variants:
                for phrase in self.variant_skipgram_index[skipgram.string]:
                    skip_matches.add_skip_match(skipgram, phrase)
        return skip_matches

    def find_candidates(self, text: dict, use_word_boundaries: bool,
                        include_variants: Union[None, bool] = None) -> List[Candidate]:
        """Find candidate fuzzy matches for a given text.

        :param text: the text object to match with phrases
        :type text: dict
        :param use_word_boundaries: use word boundaries in determining match boundaries
        :type use_word_boundaries: bool
        :param include_variants: boolean flag for whether to include phrase variants for finding matches
        :type include_variants: bool
        :return: a list of candidate matches
        :rtype: List[Candidate]
        """
        skip_matches = self.find_skipgram_matches(text, include_variants=include_variants)
        candidates = get_skipmatch_candidates(text, skip_matches, self.skipgram_threshold,
                                              max_length_variance=self.max_length_variance)
        filtered = []
        for candidate in candidates:
            if use_word_boundaries:
                adjusted_match = adjust_match_offsets(candidate.phrase.phrase_string, candidate.match_string,
                                                      text, candidate.match_start_offset, candidate.match_end_offset)
                # print("adjusting for word boundaries:", adjusted_match, candidate.match_string)
                if adjusted_match:
                    candidate.match_start_offset = adjusted_match["match_start_offset"]
                    candidate.match_end_offset = adjusted_match["match_end_offset"]
                    candidate.match_string = adjusted_match["match_string"]
                    filtered.append(candidate)
                else:
                    continue
        return filtered

    def filter_matches_by_threshold(self, matches: List[Match]) -> List[Match]:
        filtered: List[Match] = []
        for match in matches:
            # print(match.phrase.phrase_string, "\t", match.string, match.character_overlap, match.ngram_overlap, match.levenshtein_similarity)
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
                     allow_overlapping_matches: bool = True,
                     include_variants: bool = False,
                     filter_distractors: bool = False) -> List[Match]:
        """Find all fuzzy matching phrases for a given text.

        :param text: the text (string or dictionary with 'text' property) to find fuzzy matching phrases in.
        :type text: Union[str, Dict[str, str]]
        :param use_word_boundaries: use word boundaries in determining match boundaries
        :type use_word_boundaries: bool
        :param allow_overlapping_matches: boolean flag for whether to allow matches to overlap in their text ranges
        :type allow_overlapping_matches: bool
        :param include_variants: boolean flag for whether to include phrase variants for finding matches
        :type include_variants: bool
        :param filter_distractors: boolean flag for whether to remove phrase matches that better match distractors
        :type filter_distractors: bool
        :return: a list of phrases matches
        :rtype: Match
        """
        text = get_text_dict(text, ignorecase=self.ignorecase)
        if use_word_boundaries is None:
            use_word_boundaries = self.use_word_boundaries
        candidates = self.find_candidates(text, use_word_boundaries=use_word_boundaries,
                                          include_variants=include_variants)
        matches = candidates_to_matches(candidates, text, self.phrase_model)
        filtered_matches = self.filter_matches_by_threshold(matches)
        return sorted(filtered_matches, key=lambda x: x.offset)
