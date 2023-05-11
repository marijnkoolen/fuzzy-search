from __future__ import annotations
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Union

from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.match.phrase_match import PhraseMatch
from fuzzy_search.match.skip_match import SkipMatches
from fuzzy_search.phrase.phrase_model import PhraseModel
from fuzzy_search.search.searcher import FuzzySearcher
from fuzzy_search.tokenization.string import SkipGram
from fuzzy_search.tokenization.string import score_levenshtein_similarity_ratio
from fuzzy_search.tokenization.string import text2skipgrams
from fuzzy_search.tokenization.token import Doc
from fuzzy_search.tokenization.token import Token
from fuzzy_search.tokenization.token import Tokenizer


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
        self.first = text_tokens[0]
        self.last = text_tokens[-1]
        self.text_start = self.first.char_index
        self.text_end = self.last.char_index + len(self.last)
        self.text_length = self.text_end - self.text_start

    def __repr__(self):
        return f"{self.__class__.__name__}(match_type={self.match_type}, " \
               f"text_tokens={self.text_tokens}, phrase_tokens={self.phrase_tokens})"


class PartialPhraseMatch:

    def __init__(self, phrase: Phrase, token_matches: List[TokenMatch] = None, max_char_gap: int = 3,
                 max_token_gap: int = 0):
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
        self.first_text_token = None
        self.last_text_token = None
        self.first_phrase_token = None
        self.last_phrase_token = None
        if token_matches is not None:
            self.add_tokens(token_matches)

    def _update(self):
        self.text_tokens = tuple([token for match in self.token_matches for token in match.text_tokens])
        self.phrase_tokens = tuple([token for match in self.token_matches for token in match.phrase_tokens])
        self.first = self.text_tokens[0]
        self.last = self.text_tokens[-1]
        self.text_start = self.first.char_index
        self.text_end = self.last.char_index + len(self.last)
        self.text_length = self.text_end - self.text_start

    def _pop(self):
        self.token_matches.pop(0)
        self._update()

    def _check_gap(self, token_match: TokenMatch):
        token_gap = token_match.text_tokens[0].index - self.text_tokens[-1].index
        char_gap = token_match.text_tokens[0].char_index - self.text_end
        if token_gap > self.max_token_gap or char_gap > self.max_char_gap:
            self.__init__(phrase=self.phrase)

    def _push(self, token_match: TokenMatch):
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
            self.token_matches.append(token_matches)
        else:
            self.token_matches.extend(token_matches)
        self._update()


def map_text_tokens_to_phrase_tokens(partial_match: PartialPhraseMatch) -> Union[Dict[str, List[str]], None]:
    # check that each text token appears only once
    # if a text token appears multiple times, pick its best representation
    # compute the per token levenshtein similarity
    # compute the all tokens levenshtein similarity
    # compute the whole string levenshtein similarity
    text_phrase_map = defaultdict(list)
    phrase_token_set = set()
    phrase_tokens = [token.n for token in partial_match.phrase.tokens]
    for token_match in partial_match.token_matches:
        for phrase_token in token_match.phrase_tokens:
            phrase_token_set.add(phrase_token)
            for text_token in token_match.text_tokens:
                text_phrase_map[text_token].append(phrase_token)
                if phrase_token in phrase_tokens:
                    phrase_tokens.remove(phrase_token)
    if len(phrase_tokens) > 0:
        return None
    return text_phrase_map


def get_best_text_phrase_token_map(token_matches: List[TokenMatch]):
    prev_token = None
    best_phrase_tokens = []
    phrase_token_map = {}
    per_token_sim = []
    text_phrase_map = map_text_tokens_to_phrase_tokens()
    for text_token in text_phrase_map:
        best_match = None
        best_sim = 0
        for phrase_token in text_phrase_map[text_token]:
            sim = score_levenshtein_similarity_ratio(phrase_token, text_token.n)
            if sim > best_sim:
                best_match = phrase_token
                best_sim = sim
            best_phrase_tokens.append(best_match)
            per_token_sim.append(best_sim)
    for text_token in partial_match.text_tokens:
        if text_token == prev_token:
            continue
        prev_token = text_token


def get_partial_phrases(token_matches: List[TokenMatch], token_searcher: FuzzyTokenSearcher,
                        max_token_gap: int = 20):
    partial_phrase = Dict[Phrase, PartialPhraseMatch] = {}
    candidate_phrase: Dict[Phrase, List[PartialPhraseMatch]] = defaultdict(list)
    prev_token_match = None
    for token_match in token_matches:
        if prev_token_match and token_match.text_start - prev_token_match.text_end > max_token_gap:
            partial_phrase = Dict[Phrase, PartialPhraseMatch] = {}
        print(
            f"text start: {token_match.text_start: >4}\tend{token_match.text_end: >4}\t\t"
            f"text tokens: {token_match.text_tokens}\t\tphrase tokens{token_match.phrase_tokens}")
        for phrase_token_string in token_match.phrase_tokens:
            print('\tphrase_token_string:', phrase_token_string)
            for phrase_string in token_searcher.phrase_model.token_in_phrase[phrase_token_string]:
                phrase = token_searcher.phrase_model.phrase_index[phrase_string]
                print('\t\tphrase:', phrase)
                if phrase not in candidate_phrase:
                    print('\t\t\tADDING PHRASE')
                    partial_phrase[phrase] = PartialPhraseMatch([token_match], phrase)
                else:
                    print('\t\t\tADDING TOKEN')
                    partial_phrase[phrase].add_tokens(token_match)
                print('\t\t', partial_phrase[phrase].text_tokens)
                print('\t\t', partial_phrase[phrase].text_start, partial_phrase[phrase].text_end)
                print('\t\t', partial_phrase[phrase].text_length, len(phrase))
                if abs(partial_phrase[phrase].text_length - len(phrase)) <= token_searcher.config['max_length_variance']:
                    print('\t\tCANDIDATE FOUND!')
                    c
                    del partial_phrase[phrase]
                elif partial_phrase[phrase].text_length > len(phrase) + token_searcher.config['max_length_variance']:
                    print('REMOVE CANDIDATE!')
                    del partial_phrase[phrase]
        prev_token_match = token_match


def get_text_tokens(text: Union[str, Dict[str, any], Doc], tokenizer: Tokenizer = None):
    if isinstance(text, Doc):
        return text.tokens
    elif isinstance(text, list) and all(isinstance(ele, Token) for ele in text):
        return text
    elif isinstance(text, str):
        return tokenizer.tokenize(doc_text=text)
    elif isinstance(text, dict):
        return tokenizer.tokenize(doc_text=text['text'], doc_id=text['id'])
    else:
        raise TypeError(
            f'invalid text type {type(text)}, must be string, Doc or a dictionary with "text" and "id" properties')


def get_text_string(text: Union[str, Dict[str, any], Doc]) -> str:
    if isinstance(text, Doc):
        return text.text
    elif isinstance(text, list) and all(isinstance(ele, Token) for ele in text):
        return ' '.join([token.normalised_string for token in text])
    elif isinstance(text, str):
        return text
    elif isinstance(text, dict):
        return text['text']
    else:
        raise TypeError(
            f'invalid text type {type(text)}, must be string, Doc or a dictionary with "text" and "id" properties')


def get_token_skipgram_matches(text_token_skips: List[SkipGram], token_searcher: FuzzyTokenSearcher):
    token_skip_matches = SkipMatches(token_searcher.ngram_size, token_searcher.skip_size)
    for skipgram in text_token_skips:
        for phrase_token in token_searcher.token_skipgram_index[skipgram.string]:
            token_skip_matches.add_skip_match(skipgram, phrase_token)
    return token_skip_matches


class FuzzyTokenSearcher(FuzzySearcher):

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
        :param tokenizer: a tokenizer instance
        :type tokenizer: Tokenizer
        """
        super().__init__(phrase_list=phrase_list, phrase_model=phrase_model,
                         config=config, tokenizer=tokenizer)
        debug = self.config['debug']
        self.token_skipgram_index = defaultdict(set)
        self.token_num_skips = {}
        if debug > 3:
            print(f'{self.__class__.__name__}.index_phrase_model - calling index_token_skipgrams()')
        if debug > 3:
            print(f'{self.__class__.__name__}.index_phrase_model - calling index_token_skipgrams()')
        if self.phrase_model is not None:
            if debug > 3:
                print(f'{self.__class__.__name__}.index_phrase_model - calling index_token_skipgrams()')
            self.index_token_skipgrams()

    def index_token_skipgrams(self, debug: int = 0):
        debug = self._get_debug_level(debug)
        for token_string in self.phrase_model.token_in_phrase:
            if debug > 2:
                print(f'\tindex_token_skipgrams - token_string: {token_string}')
            skips = [skip for skip in text2skipgrams(token_string, self.ngram_size, skip_size=self.skip_size)]
            self.token_num_skips[token_string] = len(skips)
            for skipgram in skips:
                self.token_skipgram_index[skipgram.string].add(token_string)

    def find_skipgram_token_matches_in_text(self, text: Union[Doc, str, Dict[str, any]],
                                            debug: int = 0) -> List[TokenMatch]:
        """Find all token matches between text tokens and phrase tokens using skipgrams.

        :param text: the text object to match with tokens
        :type text: Dict[str, Union[str, int, float, list]]
        :param debug: level to show debug information
        :type debug: int
        :return: a list of matches between text tokens and phrase tokens
        :rtype: List[TokenMatch]
        """
        text_tokens = get_text_tokens(text, self.tokenizer)
        token_matches = []
        partial_matches = defaultdict(list)
        for text_token in text_tokens:
            self.find_skipgram_token_matches_for_token(text_token, partial_matches, token_matches, debug=debug)
        return token_matches

    def find_skipgram_token_matches_for_token(self, text_token: Token, partial_matches: Dict[str, List[Token]],
                                              token_matches: List[TokenMatch], debug: int = 0):
        """Find all token matches between text tokens and phrase tokens using skipgrams.

        :param text_token: a single text token to match with phrase tokens
        :type text_token: Token
        :param partial_matches: a dictionary of phrase token strings an their partial text token matches
        :type partial_matches: Dict[str, List[Token]]
        :param token_matches: a list of matches between text tokens and phrase tokens
        :type token_matches: List[TokenMatch]
        :param debug: level to show debug information
        :type debug: int
        """
        if debug > 1:
            print(f'find_skipgram_token_matches - text_token: {text_token}')
        text_token_skips = [skipgram for skipgram in text2skipgrams(text_token.normalised_string,
                                                                    self.ngram_size,
                                                                    self.skip_size)]
        token_skip_matches = get_token_skipgram_matches(text_token_skips, self)
        text_token_num_skips = len(text_token_skips)
        token_match_types = set()
        for phrase_token_match in token_skip_matches.match_offsets:
            if debug > 2:
                print(f'\tfind_skipgram_token_matches - phrase_token_match: {phrase_token_match}')
            match_type = get_token_skip_match_type(text_token.normalised_string,
                                                   text_token_num_skips,
                                                   token_skip_matches,
                                                   phrase_token_match, self)
            if debug > 2:
                print(f'\tfind_skipgram_token_matches - match_type: {match_type}')
            token_match_types.add(match_type)
            if match_type == MatchType.NONE:
                continue
            elif match_type == MatchType.FULL:
                token_match = TokenMatch(text_token, phrase_token_match, match_type)
                if debug > 2:
                    print(f'\n\tfind_skipgram_token_matches - adding full match: {token_match}\n')
                token_matches.append(token_match)
                print('NUMBER OF TOKEN MATCHES:', len(token_matches))
            elif match_type == MatchType.PARTIAL_OF_PHRASE_TOKEN:
                if debug > 2:
                    print(f'\t\tfind_skipgram_token_matches - partial_matches: {partial_matches}')
                if phrase_token_match in partial_matches:
                    if debug > 2:
                        print(phrase_token_match)
                        print(partial_matches)
                        print(partial_matches[phrase_token_match])
                    last_partial = partial_matches[phrase_token_match][-1]
                    if debug > 2:
                        print(f'\t\tfind_skipgram_token_matches - last_partial: {last_partial}')
                    if text_token.char_index - (last_partial.char_index + len(last_partial)) > 4:
                        if debug > 2:
                            print(f'\t\t\tfind_skipgram_token_matches - text_token.char_index: {text_token.char_index}')
                            print(f'\t\t\tfind_skipgram_token_matches - end of last_partial: {last_partial.char_index + len(last_partial)}')
                            print(f'\t\t\tfind_skipgram_token_matches - removing partial_match for phrase token: {phrase_token_match}')
                        del partial_matches[phrase_token_match]
                if debug > 2:
                    print(f'\t\tfind_skipgram_token_matches - adding partial "{text_token}" of phrase token "{phrase_token_match}"')
                partial_matches[phrase_token_match].append(text_token)
                if len(partial_matches[phrase_token_match]) > 1:
                    if debug > 2:
                        print(f'\t\t\tfind_skipgram_token_matches - checking if partial_matches align with phrase token: {partial_matches[phrase_token_match]}')
                    first_partial = partial_matches[phrase_token_match][0]
                    last_partial = partial_matches[phrase_token_match][-1]
                    partial_length = last_partial.char_index + len(last_partial) - first_partial.char_index
                    length_diff = partial_length - len(phrase_token_match)
                    if debug > 2:
                        print(f'\t\t\tfind_skipgram_token_matches - partial_matches: {partial_matches[phrase_token_match]}')
                        print(f'\t\t\tfind_skipgram_token_matches - partial_length: {partial_length}')
                        print(f'\t\t\tfind_skipgram_token_matches - phrase_token length: {phrase_token_match}')
                        print(f'\t\t\tfind_skipgram_token_matches - lenght_diff: {length_diff}')

                    if length_diff > 0 or abs(length_diff) <= self.config['max_length_variance']:
                        token_match = TokenMatch(partial_matches[phrase_token_match],
                                                 phrase_token_match, match_type)
                        token_matches.append(token_match)
                        print('NUMBER OF TOKEN MATCHES:', len(token_matches))
                        if debug > 2:
                            print(f'\n\tfind_skipgram_token_matches - adding full match: {token_match}')
                        partial_matches[phrase_token_match].pop(0)
            elif match_type == MatchType.PARTIAL_OF_TEXT_TOKEN:
                token_match = TokenMatch(text_token, phrase_token_match, match_type)
                token_matches.append(token_match)
        if MatchType.PARTIAL_OF_PHRASE_TOKEN not in token_match_types:
            if debug > 2:
                print(f'\tfind_skipgram_token_matches - emptying partial_matches')
            partial_matches = defaultdict(list)
        if debug > 1:
            print(partial_matches, '\n')

    def find_tokenized_matches(self, text: Union[str, List[Token]],
                               use_word_boundaries: Union[None, bool] = None,
                               allow_overlapping_matches: Union[None, bool] = None,
                               include_variants: Union[None, bool] = None,
                               filter_distractors: Union[None, bool] = None,
                               skip_exact_matching: bool = None,
                               tokenizer: Tokenizer = None,
                               debug: int = 0) -> List[PhraseMatch]:
        """Find all fuzzy matching phrases for a given text. By default, a first pass of exact matching is conducted
        to find exact occurrences of phrases. This is to speed up the fuzzy matching pass

        :param text: A tokenized text.
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
        :param tokenizer: a tokenizer instance
        :type tokenizer: Tokenizer
        :param debug: level to show debug information
        :type debug: int
        :rtype: PhraseMatch
        """
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        tokens = get_text_tokens(text, tokenizer)
        phrase_matches = []
        phrase_candidate_tokens = defaultdict(list)
        for token in tokens:
            if token.n in self.phrase_model.token_in_phrase:
                for phrase in self.phrase_model.token_in_phrase[token.n]:
                    phrase_candidate_tokens[phrase].append(token)
                phrases = self.phrase_model.token_in_phrase[token.n]
            skip_matches = self.find_skipgram_matches(token.normalised_string)

        # TODO: token match: many-to-one and one-to-many mapping of text tokens and
        #  phrase tokens (and many-to-many?)
        #  from skip matches to token matches to phrase matches
        #  1. get skip matches per text token
        #  2. filter on completish text tokens or phrase tokens
        #  3. merge sequences of text tokens that represent a single phrase token
        return phrase_matches


def get_token_skip_match_type(text_token_string: str, text_token_num_skips: int,
                              skip_matches: SkipMatches, phrase_token_match: str,
                              token_searcher: FuzzyTokenSearcher, debug: int = 0) -> MatchType:
    first = skip_matches.match_skipgrams[phrase_token_match][0]
    last = skip_matches.match_skipgrams[phrase_token_match][-1]
    overlap_start = first.offset
    overlap_end = last.offset + last.length
    num_skip_matches = len(skip_matches.match_set[phrase_token_match])
    text_token_skip_overlap = num_skip_matches / text_token_num_skips
    phrase_token_skip_overlap = num_skip_matches / token_searcher.token_num_skips[phrase_token_match]
    if text_token_skip_overlap > phrase_token_skip_overlap:
        length_variance = len(text_token_string) - (overlap_end - overlap_start)
    else:
        length_variance = len(phrase_token_match) - (overlap_end - overlap_start)
    if debug > 1:
        print(f"get_token_skip_match_type - first:", first)
        print(f"get_token_skip_match_type - last:", last)
        print(f"get_token_skip_match_type - overlap_start:", overlap_start)
        print(f"get_token_skip_match_type - overlap_end:", overlap_end)
        print(f"get_token_skip_match_type - num_skip_matches:", num_skip_matches)
        print(f"get_token_skip_match_type - text_token_skip_overlap:", text_token_skip_overlap)
        print(f"get_token_skip_match_type - phrase_token_skip_overlap:", phrase_token_skip_overlap)
        print(f"get_token_skip_match_type - length_variance:", length_variance)
    if text_token_skip_overlap < token_searcher.config['skipgram_threshold'] and \
            phrase_token_skip_overlap < token_searcher.config['skipgram_threshold']:
        match_type = MatchType.NONE
        if debug > 1:
            print(f"get_token_skip_match_type - below skipgram thresholds, match_type:", match_type)
    elif length_variance > token_searcher.config['max_length_variance']:
        match_type = MatchType.NONE
        if debug > 1:
            print(f"get_token_skip_match_type - above max length variance, match_type:", match_type)
    elif abs(len(text_token_string) - len(phrase_token_match)) < token_searcher.config['max_length_variance']:
        match_type = MatchType.FULL
        if debug > 1:
            print(f"get_token_skip_match_type - text and phrase tokens equal length, match_type:", match_type)
    elif len(text_token_string) < len(phrase_token_match):
        match_type = MatchType.PARTIAL_OF_PHRASE_TOKEN
        if debug > 1:
            print(f"get_token_skip_match_type - phrase token longer than text token, match_type:", match_type)
    else:
        match_type = MatchType.PARTIAL_OF_TEXT_TOKEN
        if debug > 1:
            print(f"get_token_skip_match_type - text token longer than phrase token, match_type:", match_type)
    return match_type
