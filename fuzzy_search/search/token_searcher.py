from __future__ import annotations
import time
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
from fuzzy_search.tokenization.vocabulary import Vocabulary


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
        return f"{self.__class__.__name__}(phrase={self.phrase}, token_matches={self.token_matches}, " \
               f"text_tokens={self.text_tokens}, phrase_tokens={self.phrase_tokens}," \
               f", missing_tokens={self.missing_tokens})"

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
        self.first = self.text_tokens[0]
        self.last = self.text_tokens[-1]
        self.text_start = self.first.char_index
        self.text_end = self.last.char_index + len(self.last)
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


def get_partial_phrases(token_matches: List[TokenMatch], token_searcher: FuzzyTokenSearcher,
                        max_char_gap: int = 20, debug: int = 0):
    partial_phrase: Dict[Phrase, PartialPhraseMatch] = {}
    partial_phrases = {}
    candidate_phrase: Dict[Phrase, List[PartialPhraseMatch]] = defaultdict(list)
    prev_token_match = None
    for token_match in token_matches:
        if prev_token_match and token_match.text_start - prev_token_match.text_end > max_char_gap:
            partial_phrase: Dict[Phrase, PartialPhraseMatch] = {}
        if debug > 1:
            print(
                f"text start: {token_match.text_start: >4}\tend{token_match.text_end: >4}\t\t"
                f"text tokens: {token_match.text_tokens}\t\tphrase tokens{token_match.phrase_tokens}")
        for phrase_token_string in token_match.phrase_tokens:
            if debug > 1:
                print('\tphrase_token_string:', phrase_token_string)
            for phrase_string in token_searcher.phrase_model.token_in_phrase[phrase_token_string]:
                # phrase = token_searcher.phrase_model.phrase_index[phrase_string]
                phrase = token_searcher.phrase_model.get_phrase(phrase_string)
                if phrase is None:
                    continue
                # print('phrase:', phrase)
                # print('phrase.tokens:', phrase.tokens)
                if debug > 1:
                    print('\t\tphrase:', phrase)
                if phrase not in candidate_phrase:
                    if debug > 1:
                        print('\t\t\tADDING PHRASE')
                    partial = PartialPhraseMatch(phrase, [token_match])
                    # partial_phrase[phrase] = partial
                    candidate_phrase[phrase].append(partial)
                else:
                    added = False
                    for partial in candidate_phrase[phrase]:
                        # check if token extends existing partial match
                        if token_match.text_start - partial.text_end > max_char_gap:
                            # print('skipping partial match:', token_match.text_start, partial.text_end)
                            continue
                        if partial.text_end < token_match.text_start and \
                                any([pt for pt in token_match.phrase_tokens if pt in partial.missing_tokens]):
                            partial.add_tokens([token_match])
                            if debug > 1:
                                print('\t\t\tADDING TOKEN')
                                print('\t\t', partial.text_tokens)
                                print('\t\t', partial.text_start, partial.text_end)
                                print('\t\t', partial.text_length, len(phrase))
                            added = True
                    if not added:
                        partial = PartialPhraseMatch(phrase, [token_match])
                        if debug > 1:
                            print('\t\t\tADDING PARTIAL TO PHRASE')
                            print('\t\t', partial.text_tokens)
                            print('\t\t', partial.text_start, partial.text_end)
                            print('\t\t', partial.text_length, len(phrase))
                        candidate_phrase[phrase].append(partial)
        prev_token_match = token_match
    for phrase in candidate_phrase:
        remove_phrases = []
        for partial in candidate_phrase[phrase]:
            if debug > 1:
                print('\tCHECKING PARTIAL CANDIDATE')
                print('\t\t', partial.text_tokens)
                print('\t\t', partial.text_start, partial.text_end)
                print('\t\t', partial.text_length, len(phrase))
            if abs(partial.text_length - len(phrase)) > token_searcher.config['max_length_variance']:
                if debug > 0:
                    print('\t\tREMOVE CANDIDATE!')
                remove_phrases.append(partial)
                candidate_phrase[phrase].remove(partial)
                # candidate_phrase.append(partial)
                # del partial
            # elif partial.text_length > len(phrase) + token_searcher.config['max_length_variance']:
            else:
                if debug > 0:
                    print('\t\tCANDIDATE FOUND!')
                # del partial[phrase]
    return candidate_phrase


def get_tokenized_doc(text: Union[str, Dict[str, any], Doc], tokenizer: Tokenizer) -> Doc:
    if isinstance(text, Doc):
        return text
    elif isinstance(text, dict):
        return tokenizer.tokenize(text['text'], doc_id=text['id'])
    elif isinstance(text, str):
        return tokenizer.tokenize(text)
    else:
        raise TypeError(f"text must be str, dict (with 'text' and 'id' properties) or Doc, not {type(text)}")


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


def get_token_skipgram_matches(text_token_skips: List[SkipGram], token_searcher: FuzzyTokenSearcher,
                               debug: int = 0):
    token_skip_matches = SkipMatches(token_searcher.ngram_size, token_searcher.skip_size)
    for skipgram in text_token_skips:
        for phrase_token in token_searcher.token_skipgram_index[skipgram.string]:
            token_skip_matches.add_skip_match(skipgram, phrase_token)
    return token_skip_matches


class FuzzyTokenSearcher(FuzzySearcher):

    def __init__(self, phrase_list: List[any] = None, phrase_model: Union[Dict[str, any], PhraseModel] = None,
                 config: Union[None, Dict[str, Union[str, int, float]]] = None, tokenizer: Tokenizer = None,
                 vocabulary: Vocabulary = None, max_char_gap: int = 20, max_token_gap: int = 1):
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
        self.max_token_gap = max_token_gap
        self.max_char_gap = max_char_gap
        self.vocabulary = vocabulary
        if debug > 3:
            print(f'{self.__class__.__name__}.index_phrase_model - calling index_token_skipgrams()')
        if debug > 3:
            print(f'{self.__class__.__name__}.index_phrase_model - calling index_token_skipgrams()')
        if self.phrase_model is not None:
            if debug > 3:
                print(f'{self.__class__.__name__}.index_phrase_model - calling index_token_skipgrams()')
            self.index_token_skipgrams()

    def configure(self, config: Dict[str, any]):
        for prop in config:
            if hasattr(self, prop):
                self.__setattr__(prop, config[prop])

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
            if self.vocabulary and self.vocabulary.has_term(text_token):
                self.find_vocabulary_token_matches_for_token(text_token, token_matches, debug=debug)
                continue
            if debug > 0:
                print('\nfind_skipgram_token_matches_in_text - text_token:', text_token)
            self.find_skipgram_token_matches_for_token(text_token, partial_matches, token_matches, debug=debug)
        return token_matches

    def find_vocabulary_token_matches_for_token(self, text_token: Token, token_matches: List[TokenMatch],
                                                debug: int = 0):
        if text_token.n not in self.phrase_model.token_in_phrase:
            return None
        for phrase_string in self.phrase_model.token_in_phrase[text_token.n]:
            phrase = self.phrase_model.get_phrase(phrase_string)
            if isinstance(phrase, Phrase):
                phrase_tokens = [token for token in phrase.tokens if token.n == text_token.n]
                token_match = TokenMatch(text_tokens=text_token, phrase_tokens=phrase_tokens,
                                         match_type=MatchType.FULL)
                token_matches.append(token_match)

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
        if debug > 0:
            print(f'\nfind_skipgram_token_matches - text_token: {text_token}')
        text_token_skips = [skipgram for skipgram in text2skipgrams(text_token.normalised_string,
                                                                    self.ngram_size,
                                                                    self.skip_size)]
        token_skip_matches = get_token_skipgram_matches(text_token_skips, self)
        if debug > 0:
            print(f'find_skipgram_token_matches - number of phrase matches: '
                  f'{len(token_skip_matches.match_start_offsets)}')
            print()
        text_token_num_skips = len(text_token_skips)
        token_match_types = set()
        for phrase_token_match in token_skip_matches.match_start_offsets:
            if debug > 1:
                print(f'\n\tfind_skipgram_token_matches - phrase_token_match: {phrase_token_match}')
                print(f'\tfind_skipgram_token_matches - phrase_token_match: {phrase_token_match}')
                print(f'\tfind_skipgram_token_matches - number of skipgram matches: '
                      f'{len(token_skip_matches.match_start_offsets[phrase_token_match])}')
                print(f'\tfind_skipgram_token_matches - skipgram matches: ',
                      token_skip_matches.match_skipgrams[phrase_token_match][0].start_offset,
                      token_skip_matches.match_skipgrams[phrase_token_match][-1].end_offset,
                      )
            match_type = get_token_skip_match_type(text_token.normalised_string,
                                                   text_token_num_skips,
                                                   token_skip_matches,
                                                   phrase_token_match, self, debug=debug)
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
                if debug > 0:
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
                        if debug > 0:
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

    def _pick_best_candidates(self, doc: Doc, candidate_phrases: Dict[Phrase, List[PartialPhraseMatch]],
                              debug: int = 0):
        if debug > 0:
            print('\nfind_phrase_matches - listing partial matches:')
            for phrase in candidate_phrases:
                print('\tfind_phrase_matches - phrase:', phrase)
                for partial in candidate_phrases[phrase]:
                    print('\t\tfind_phrase_matches - partial:', partial)
        phrase_at_offset: Dict[int, PartialPhraseMatch] = {}
        for phrase in candidate_phrases:
            for pp in candidate_phrases[phrase]:
                pp.match_string = doc.text[pp.text_start:pp.text_end]
                # the match_string is only the text tokens, not the text between
                # start and end offsets: if there are non-matching tokens in between,
                # the Levenshtein score should be computed only on the matching text
                # tokens.
                pp.match_string = ' '.join([token.n for token in pp.text_tokens])
                if debug > 0:
                    print('\nfind_phrase_matches - pp:', pp.phrase.phrase_string)
                    print('\tfind_phrase_matches - pp.match_string:', pp.match_string)
                if abs(len(pp.match_string) - len(pp.phrase.phrase_string)) > self.max_length_variance:
                    continue
                pp.levenshtein_score = score_levenshtein_similarity_ratio(pp.phrase.phrase_string, pp.match_string)
                if debug > 1:
                    print('\tfind_phrase_matches - pp.levenshtein_score:', pp.levenshtein_score)
                if pp.levenshtein_score < self.config['levenshtein_threshold']:
                    if debug > 0:
                        print(f'\t\tfind_phrase_matches - phrase below score threshold {pp.text_start}:', pp.phrase.phrase_string)
                    continue
                if pp.text_start in phrase_at_offset:
                    if debug > 0:
                        print(f'\t\tfind_phrase_matches - next phrase at offset {pp.text_start}:', pp.phrase.phrase_string)
                    if pp.levenshtein_score > phrase_at_offset[pp.text_start].levenshtein_score:
                        if debug > 0:
                            print(f'\t\tfind_phrase_matches - replacing phrase at offset {pp.text_start}:', pp.phrase.phrase_string)
                        phrase_at_offset[pp.text_start] = pp
                else:
                    if debug > 0:
                        print(f'\t\tfind_phrase_matches - first phrase at offset {pp.text_start}:', pp.phrase.phrase_string)
                    phrase_at_offset[pp.text_start] = pp
        phrases = []
        for partial_phrase in sorted(phrase_at_offset.values(), key=lambda x: x.text_start):
            phrase = PhraseMatch(partial_phrase.phrase, partial_phrase.phrase, partial_phrase.match_string,
                                 partial_phrase.text_start, text_id=doc.id,
                                 levenshtein_similarity=partial_phrase.levenshtein_score,
                                 match_label=list(partial_phrase.phrase.label_set))
            phrases.append(phrase)
        return phrases

    def find_matches(self, text: Union[Doc, str, Dict[str, any]],
                     debug: int = 0) -> List[PhraseMatch]:
        """Find all fuzzy matching phrases for a given text using token-based searching.
        The `FuzzyTokenSearcher` turns the phrases and the target text into lists of word
        tokens (the tokenizer is configurable) and uses character skip grams to identify
        candidate phrase tokens matching tokens in the text. It then uses token sequences
        to identify fuzzy matches.

        This speeds up the search (especially for the default settings `ngram_size=2` and
        `skip_size=2`) at the cost of slightly less exhaustive search.

        :param text: A tokenized text.
        :type text: Union[str, Dict[str, str]]
        :param debug: level to show debug information
        :type debug: int
        :return: a list of phrase matches
        :rtype: List[PartialPhraseMatch]
        """
        start = time.time()
        doc = get_tokenized_doc(text, self.tokenizer)
        token_matches = self.find_skipgram_token_matches_in_text(doc, debug=debug)
        if debug > 0:
            print('find_phrase_matches - number of token_matches:', len(token_matches))
        if debug > 1:
            print('find_phrase_matches - number of token_matches:', len(token_matches))
            step1 = time.time()
            print(f'step 1 took: {step1 - start: >.2f} seconds')
        if debug > 1:
            for tm in token_matches:
                print('find_phrase_matches - token_match:', tm)
        candidate_phrases = get_partial_phrases(token_matches, self)
        if debug > 1:
            print('find_phrase_matches - number of candidate phrases:', len(candidate_phrases))
            print('find_phrase_matches - number of partial phrases:', sum([len(candidate_phrases[phrase]) for phrase in candidate_phrases]))
            step2 = time.time()
            print(f'step 2 took: {step2 - step1: >.2f} seconds')
        phrases = self._pick_best_candidates(doc, candidate_phrases, debug=debug)
        if debug > 1:
            step3 = time.time()
            print(f'step 3 took: {step3 - step2: >.2f} seconds')
        return phrases


def get_token_skip_match_type(text_token_string: str, text_token_num_skips: int,
                              skip_matches: SkipMatches, phrase_token_match: str,
                              token_searcher: FuzzyTokenSearcher, debug: int = 0) -> MatchType:
    first = skip_matches.match_skipgrams[phrase_token_match][0]
    last = skip_matches.match_skipgrams[phrase_token_match][-1]
    overlap_start = first.start_offset
    overlap_end = last.start_offset + last.length
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
