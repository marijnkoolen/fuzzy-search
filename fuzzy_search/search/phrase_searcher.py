import time
from typing import Dict, List, Union

from fuzzy_search.phrase.phrase_model import PhraseModel
from fuzzy_search.match.phrase_match import PhraseMatch
from fuzzy_search.match.phrase_match import Candidate
from fuzzy_search.match.phrase_match import adjust_match_offsets
from fuzzy_search.match.phrase_match import candidates_to_matches
from fuzzy_search.match.phrase_match import filter_matches_by_overlap
from fuzzy_search.match.skip_match import get_skipmatch_candidates
from fuzzy_search.match.exact_match import index_known_word_offsets
from fuzzy_search.match.exact_match import search_exact_phrases
from fuzzy_search.search.searcher import FuzzySearcher
from fuzzy_search.search.token_searcher import FuzzyTokenSearcher
from fuzzy_search.tokenization.string import score_levenshtein_similarity_ratio
from fuzzy_search.tokenization.token import Tokenizer
from fuzzy_search.tokenization.token import Doc


def get_text_dict(text: Union[str, dict, Doc], ignorecase: bool = False) -> dict:
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
        text = {"text": text, "id": None, 'text_lower': text.lower()}
    elif isinstance(text, Doc):
        text = {'text': text.text, 'id': text.id}
    if "id" not in text:
        text["id"] = None
    return text


def combine_fuzzy_and_exact_matches(filtered_matches: List[PhraseMatch],
                                    exact_matches: List[PhraseMatch],
                                    debug: int = 0):
    combined_matches = [em for em in exact_matches]
    exact_phrases = {(em.offset, em.phrase.phrase_string): em for em in exact_matches}
    if debug > 2:
        print('combine_fuzzy_and_exact_matches - exact_phrases.keys():', exact_phrases.keys())
    for fm in filtered_matches:
        if (fm.offset, fm.phrase.phrase_string) in exact_phrases:
            if debug > 2:
                print('skipping fuzzy match because there is a better exact match:', fm)
            continue
        else:
            combined_matches.append(fm)
    return sorted(combined_matches, key=lambda m: m.offset)


class FuzzyPhraseSearcher(FuzzySearcher):

    def __init__(self, phrase_list: List[any] = None,
                 phrase_model: Union[Dict[str, any], List[Dict[str, any]], PhraseModel] = None,
                 config: Union[None, Dict[str, Union[str, int, float]]] = None,
                 tokenizer: Tokenizer = None,
                 token_searcher: FuzzyTokenSearcher = None):
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
        :param tokenizer: a tokenizer instance
        :type tokenizer: Tokenizer
        :param token_searcher: a fuzzy token searcher instance (using the same phrase model and config)
        :type token_searcher: FuzzyTokenSearcher
        """
        super().__init__(phrase_list=phrase_list, phrase_model=phrase_model,
                         config=config, tokenizer=tokenizer)
        if token_searcher is None:
            token_searcher = FuzzyTokenSearcher(phrase_model=self.phrase_model,
                                                config=self.config, tokenizer=self.tokenizer)
        self.token_searcher = token_searcher

    def find_candidates(self, text: dict, use_word_boundaries: bool,
                        include_variants: Union[None, bool] = None,
                        known_word_start_offset: Dict[int, Dict[str, any]] = None,
                        debug: int = 0) -> List[Candidate]:
        """Find candidate fuzzy matches for a given text.

        :param text: the text object to match with phrases
        :type text: dict
        :param use_word_boundaries: use word boundaries in determining match boundaries
        :type use_word_boundaries: bool
        :param include_variants: boolean flag for whether to include phrase variants for finding matches
        :type include_variants: bool
        :param known_word_start_offset: a dictionary of known words and their text offsets based on exact matches
        :type known_word_start_offset: Dict[int, Dict[str, any]]
        :param debug: level to show debug information
        :type debug: int
        :return: a list of candidate matches
        :rtype: List[Candidate]
        """
        skip_matches = self.find_skipgram_matches(text, include_variants=include_variants,
                                                  known_word_start_offset=known_word_start_offset)
        candidates = get_skipmatch_candidates(text, skip_matches, self.skipgram_threshold, self.phrase_model,
                                              max_length_variance=self.max_length_variance,
                                              ignorecase=self.ignorecase, debug=debug)
        if debug > 2:
            print('find_candidates - candidates:')
            for candidate in candidates:
                print(f"\t{candidate.phrase}\t-\t{candidate.match_string}")
        filtered = []
        use_word_boundaries = use_word_boundaries if use_word_boundaries is not None else self.use_word_boundaries
        if not use_word_boundaries:
            filtered = candidates
        else:
            if debug > 1:
                print('find_candidates - start word boundary filtering candidates')
            for candidate in candidates:
                if debug > 3:
                    print()
                    print('find_candidates - candidate:', candidate)
                adjusted_match = adjust_match_offsets(candidate.phrase.phrase_string, candidate.match_string,
                                                      text, candidate.match_start_offset, candidate.match_end_offset,
                                                      self.punctuation, debug=debug)
                if debug > 3:
                    print("find_candidates - adjusted_match:", adjusted_match)
                if not adjusted_match:
                    if debug > 2:
                        print(f"find_candidates - removing candidate '{candidate.phrase}'\t-\t'{candidate.match_string}'")
                    continue
                candidate.match_start_offset = adjusted_match["match_start_offset"]
                candidate.match_end_offset = adjusted_match["match_end_offset"]
                candidate.match_string = adjusted_match["match_string"]
                if debug > 3:
                    print("find_candidates - new match string:", candidate.match_string)
                filtered.append(candidate)
        if debug > 1:
            print('find_candidates - returning word boundary filtered candidates:')
            for candidate in filtered:
                print(f"\t{candidate.phrase.phrase_string}\t-\t{candidate.match_string}")
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

    def find_matches(self, text: Union[str, Dict[str, str], Doc],
                     use_word_boundaries: Union[None, bool] = None,
                     allow_overlapping_matches: Union[None, bool] = None,
                     include_variants: Union[None, bool] = None,
                     filter_distractors: Union[None, bool] = None,
                     skip_exact_matching: bool = None,
                     first_best: bool = False,
                     debug: int = 0) -> List[PhraseMatch]:
        """Find all fuzzy matching phrases for a given text. By default, a first pass of exact matching is conducted
        to find exact occurrences of phrases. This is to speed up the fuzzy matching pass

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
        :param skip_exact_matching: boolean flag whether to skip the exact matching step
        :type skip_exact_matching: bool
        :return: a list of phrases matches
        :type first_best: bool
        :return: whether to return only the first match with the best score, or all matches
        :param debug: level to show debug information
        :type debug: int
        :rtype: PhraseMatch
        """
        time_step = make_step_timer()
        if debug > 0:
            print('find_matches - getting text dict')
        if self.phrase_model is None:
            raise ValueError("No phrase model indexed")
        text = get_text_dict(text, ignorecase=self.ignorecase)
        if use_word_boundaries is None:
            use_word_boundaries = self.use_word_boundaries
        if skip_exact_matching is None:
            skip_exact_matching = self.skip_exact_matching
        if not skip_exact_matching:
            if debug > 0:
                time_step()
                print("find_matches - running exact matching")
            exact_matches = self.find_exact_matches(text, use_word_boundaries=use_word_boundaries,
                                                    include_variants=include_variants)
            known_word_start_offset = index_known_word_offsets(exact_matches)
        else:
            if debug > 0:
                time_step()
                print("find_matches - skipping exact matching")
            exact_matches = []
            known_word_start_offset = {}
        if debug > 0:
            time_step()
            print('find_matches - number of exact matches:', len(exact_matches))
        candidates = self.find_candidates(text, use_word_boundaries=use_word_boundaries,
                                          include_variants=include_variants,
                                          known_word_start_offset=known_word_start_offset, debug=debug)
        if debug > 0:
            print('find_matches - number of candidates received from find_candidates:', len(candidates))
        if debug > 0:
            time_step()
        if debug > 2:
            for candidate in candidates:
                print(f"\t{candidate.phrase.phrase_string}\t-\t{candidate.match_string}")
        matches = candidates_to_matches(candidates, text, self.phrase_model, ignorecase=self.ignorecase)
        if debug > 0:
            time_step()
            print('find_macthes - number of matches:', len(matches))
        if debug > 2:
            print('find_macthes - matches:')
            for match in matches:
                print(f"\tmatch phrase_string: {match.phrase.phrase_string}\tmatch_string: {match.string}")
        filtered_matches = self.filter_matches_by_threshold(matches)
        if filter_distractors is None:
            filter_distractors = self.filter_distractors
        if filter_distractors:
            filtered_matches = self.filter_matches_by_distractors(filtered_matches)
            if debug > 1:
                print('find_macthes - number of matches after filtering by distractors:', len(matches))
        if allow_overlapping_matches is None:
            allow_overlapping_matches = self.allow_overlapping_matches
        if debug > 1:
            print(len(filtered_matches), len(exact_matches))
            print([fm.levenshtein_similarity for fm in filtered_matches])
            print([em.levenshtein_similarity for em in exact_matches])
            for fm in filtered_matches:
                print(fm)
        filtered_matches = combine_fuzzy_and_exact_matches(filtered_matches, exact_matches)
        if debug > 1:
            print('find_macthes - number of matches after combining fuzzy and exact matches:', len(filtered_matches))
        filtered_matches = self.filter_matches_by_offset_threshold(filtered_matches, debug=debug)
        if debug > 1:
            print('find_macthes - number of matches after filtering by offset threshold:', len(filtered_matches))
        if not allow_overlapping_matches:
            filtered_matches = filter_matches_by_overlap(filtered_matches, first_best=first_best, debug=debug)
            if debug > 1:
                print('find_macthes - number of matches after filtering by overlap:', len(filtered_matches))
        # print(exact_matches)
        if debug > 0:
            time_step()
            print('find_matches - filtered_matches:', filtered_matches)
        return sorted(filtered_matches, key=lambda x: (x.text_id, x.offset, x.offset + len(x.string)))

    def find_exact_matches(self, text: Union[str, Dict[str, str]],
                           use_word_boundaries: Union[None, bool] = None,
                           include_variants: Union[None, bool] = None,
                           debug: int = 0) -> List[PhraseMatch]:
        """Find all fuzzy matching phrases for a given text.

        :param text: the text (string or dictionary with 'text' property) to find fuzzy matching phrases in.
        :type text: Union[str, Dict[str, str]]
        :param use_word_boundaries: use word boundaries in determining match boundaries
        :type use_word_boundaries: Union[None, bool]
        :param include_variants: boolean flag for whether to include phrase variants for finding matches
        :type include_variants: Union[None, bool]
        :param debug: level to show debug information
        :type debug: int
        :return: a list of phrases matches
        :rtype: PhraseMatch
        """
        exact_matches: List[PhraseMatch] = []
        text = get_text_dict(text, ignorecase=self.ignorecase)
        if use_word_boundaries is None:
            use_word_boundaries = self.use_word_boundaries
        if include_variants is None:
            include_variants = self.include_variants
        if debug > 0:
            print('find_exact_matches - use_word_boundaries:', use_word_boundaries)
            print('find_exact_matches - include_variants:', include_variants)
        for exact_match in search_exact_phrases(self.phrase_model, text, use_word_boundaries=use_word_boundaries,
                                                include_variants=include_variants, debug=debug):
            exact_matches.append(exact_match)
        return exact_matches


def make_step_timer():

    first_step = time.time()
    prev_step = first_step

    def time_step():
        nonlocal prev_step
        curr_step = time.time()
        took = curr_step - prev_step
        prev_step = curr_step
        print(f'\tstep took {took: >.2f} seconds, total: {curr_step - first_step: >.2f}')
        return took

    return time_step

