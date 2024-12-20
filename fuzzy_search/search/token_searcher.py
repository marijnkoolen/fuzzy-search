from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Iterable, List, Union

from Levenshtein import distance
from Levenshtein import ratio as score_ratio

from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.match.phrase_match import PhraseMatch
from fuzzy_search.match.phrase_match import MatchType
from fuzzy_search.match.phrase_match import TokenMatch
from fuzzy_search.match.phrase_match import PartialPhraseMatch
from fuzzy_search.match.phrase_match import copy_partial_match
from fuzzy_search.match.skip_match import SkipMatches
from fuzzy_search.phrase.phrase_model import PhraseModel
from fuzzy_search.search.searcher import FuzzySearcher
from fuzzy_search.tokenization.string import token2skipgrams
from fuzzy_search.tokenization.token import Doc
from fuzzy_search.tokenization.token import Token
from fuzzy_search.tokenization.token import Tokenizer
from fuzzy_search.tokenization.vocabulary import Vocabulary


def map_text_tokens_to_phrase_tokens(partial_match: PartialPhraseMatch) -> Union[Dict[str, List[str]], None]:
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


def get_tokenized_doc(text: Union[str, Dict[str, any], Doc], tokenizer: Tokenizer) -> Doc:
    if isinstance(text, Doc):
        return text
    elif isinstance(text, dict):
        return tokenizer.tokenize_doc(text['text'], doc_id=text['id'])
    elif isinstance(text, str):
        return tokenizer.tokenize_doc(text)
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


def has_max_start_offset(phrase: Phrase):
    return phrase.max_start_offset is not None and phrase.max_start_offset != -1


def has_max_end_offset(phrase: Phrase):
    return phrase.max_end_offset is not None and phrase.max_end_offset != -1


class FuzzyTokenSearcher(FuzzySearcher):

    def __init__(self, phrase_list: List[any] = None,
                 phrase_model: Union[Dict[str, any], List[Dict[str, any]], PhraseModel] = None,
                 config: Union[None, Dict[str, Union[str, int, float]]] = None, tokenizer: Tokenizer = None,
                 vocabulary: [Vocabulary, List[str]] = None,
                 index_vocabulary_pairs: bool = True,
                 max_char_gap: int = 20, max_token_gap: int = 1, debug: int = 0):
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
        if debug is None and self.config['debug'] is not None:
            debug = self.config['debug']
        self.debug = debug
        if 'pad_token' not in self.config:
            self.config['pad_token'] = False
        self.token_skipgram_index = defaultdict(set)
        self.token_num_skips = {}
        self.max_token_gap = max_token_gap
        self.max_char_gap = max_char_gap
        if 'max_token_length_variance' not in self.config:
            self.config['max_token_length_variance'] = self.config['max_length_variance']
        self.vocabulary = Vocabulary()
        self.match_pairs = set()
        self.text_phrase_term_pairs = {
            'match': set(),
            'distractor': set()
        }
        self.distractor_pairs = set()
        self.partial_count = 0
        self.vocabulary_skipgram_matches = defaultdict(SkipMatches)
        if self.phrase_model is not None:
            if self.debug > 3:
                print(f'{self.__class__.__name__}.index_phrase_model - calling index_phrase_token_skipgrams()')
            self.index_phrase_token_skipgrams()
        if vocabulary is not None:
            self.add_vocabulary(vocabulary)
        if self.phrase_model is not None:
            phrase_tokens = list(set([pt for pt in self.phrase_model.token_in_phrase]))
            self.add_vocabulary(phrase_tokens)
        if index_vocabulary_pairs is True:
            match_pairs, distractor_pairs = self.find_vocabulary_text_phrase_term_pairs()
            self.index_text_phrase_term_pairs(match_pairs, 'match')
            self.index_text_phrase_term_pairs(distractor_pairs, 'distractor')
        self.add_vocabulary_skipgram_matches()
        # self.term_dist = defaultdict(int)

    @staticmethod
    def terms_to_string(terms: Union[str, List[str], tuple]):
        if isinstance(terms, str):
            pass
        elif isinstance(terms, Iterable):
            terms = ' '.join(terms)
        if isinstance(terms, str) is False:
            raise TypeError(f"terms '{terms}' is not properly converted to string")
        return terms

    @staticmethod
    def terms_to_tuple(terms: Union[str, List[str], tuple]):
        if isinstance(terms, str):
            terms = tuple([terms])
        elif isinstance(terms, list):
            terms = tuple(terms)
        return tuple(terms)

    def terms_to_id_tuple(self, terms: Union[str, tuple]):
        # print(terms, type(terms))
        terms = self.terms_to_tuple(terms)
        if any([self.vocabulary.has_term(tt) is False for tt in terms]):
            return None
        return tuple([self.vocabulary.term_id[tt] for tt in terms])

    def index_text_phrase_term_pairs(self, text_phrase_term_pairs, pair_type: str):
        for text_terms, phrase_terms in text_phrase_term_pairs:
            try:
                self.index_text_phrase_term_pair(text_terms=text_terms, phrase_terms=phrase_terms, pair_type=pair_type)
            except TypeError:
                print(text_terms, phrase_terms)
                raise

    def index_text_phrase_term_pair(self, text_terms: Union[str, tuple],
                                    phrase_terms: Union[str, tuple],
                                    pair_type: str):
        text_term_ids = self.terms_to_id_tuple(text_terms)
        phrase_term_ids = self.terms_to_id_tuple(phrase_terms)
        if text_term_ids is None or phrase_term_ids is None:
            return None
        self.text_phrase_term_pairs[pair_type].add((text_term_ids, phrase_term_ids))
        if pair_type == 'distractor':
            term_string = self.terms_to_string(text_terms)
            if term_string not in self.vocabulary_skipgram_matches:
                term_token = Token(string=term_string, index=0, char_index=0, char_end_index=len(term_string))
                self.vocabulary_skipgram_matches[term_string] = get_token_skipgram_matches(term_token, self)
            self.vocabulary_skipgram_matches[term_string].match_type[phrase_terms] = MatchType.NONE

    def index_distractor_pair(self, text_terms: Union[str, tuple],
                              phrase_terms: Union[str, tuple]):
        self.index_text_phrase_term_pair(text_terms, phrase_terms, 'distractor')

    def find_vocabulary_text_phrase_term_pairs(self):
        if self.vocabulary is None:
            return None
        distractor_pairs = set()
        match_pairs = set()
        for term in self.vocabulary:
            # print(f"TokenSearcher.index_distractor_pairs - term: {term}")
            term_token = Token(string=term, index=0, char_index=0, char_end_index=len(term))
            term_skip_matches = get_token_skipgram_matches(term_token, self, debug=0)
            # print(term_skip_matches.match_start_offsets)
            for ptm in term_skip_matches.match_start_offsets:
                # self.term_dist[(term, ptm)] = distance(term, ptm)
                if self.debug > 1:
                    print(f"   ptm: {ptm}")
                    print(f"    is_distractor: {is_distractor(term, ptm, debug=self.debug)}")
                if not self.vocabulary.has_term(ptm):
                    print(f" type '{ptm}': {type(ptm)}")
                    raise ValueError(f"phrase token '{ptm}' is not in vocabulary.")
                if is_distractor(term, ptm):
                    distractor_pairs.add(((term,), (ptm,)))
                else:
                    match_pairs.add(((term,), (ptm,)))
        return match_pairs, distractor_pairs

    def has_text_phrase_term_pair(self, text_terms: Union[str, tuple], phrase_terms: Union[str, tuple],
                                  pair_type: str):
        text_term_ids = self.terms_to_id_tuple(text_terms)
        phrase_term_ids = self.terms_to_id_tuple(phrase_terms)
        if text_term_ids is None or phrase_term_ids is None:
            return False
        if self.debug > 1:
            print(self.match_pairs)
        return (text_term_ids, phrase_term_ids) in self.text_phrase_term_pairs[pair_type]

    def has_match_pair(self, text_terms: Union[str, tuple], phrase_terms: Union[str, tuple]):
        return self.has_text_phrase_term_pair(text_terms, phrase_terms, pair_type='match')

    def has_distractor_pair(self, text_terms: Union[str, tuple], phrase_terms: Union[str, tuple]):
        return self.has_text_phrase_term_pair(text_terms, phrase_terms, pair_type='distractor')

    def add_vocabulary(self, vocab: Union[List[str], Vocabulary]):
        if isinstance(vocab, Vocabulary):
            if len(self.vocabulary) == 0:
                self.vocabulary = vocab
            else:
                # print("adding terms from vocabulary", [term for term in vocab])
                self.vocabulary.add_terms([term for term in vocab], reset_index=False)
        elif isinstance(vocab, list):
            # print("adding terms from list", [term for term in vocab])
            self.vocabulary.add_terms(vocab, reset_index=False)
            # print(self.vocabulary.term_id)
        else:
            raise TypeError("Vocabulary 'vocab' must be a list of strings or a Vocabulary instance.")

    def configure(self, config: Dict[str, any]):
        for prop in config:
            if hasattr(self, prop):
                self.__setattr__(prop, config[prop])

    def index_phrase_token_skipgrams(self, debug: int = 0):
        debug = self._get_debug_level(debug)
        for token_string in self.phrase_model.token_in_phrase:
            if debug > 2:
                print(f'\tindex_phrase_token_skipgrams - token_string: {token_string}')
            skips = [skip for skip in token2skipgrams(token_string, self.ngram_size, skip_size=self.skip_size,
                                                      pad_token=self.config['pad_token'])]
            self.token_num_skips[token_string] = len(skips)
            for skipgram in skips:
                self.token_skipgram_index[skipgram.string].add(token_string)

    def add_vocabulary_skipgram_matches(self):
        for term in self.vocabulary.term_id:
            term_token = Token(string=term, index=0, char_index=0, char_end_index=len(term))
            # self.config['skipgram_threshold'] += 0.2
            self.vocabulary_skipgram_matches[term] = get_token_skipgram_matches(term_token, self)
            # self.config['skipgram_threshold'] -= 0.2
            ptms = list(self.vocabulary_skipgram_matches[term].match_start_offsets.keys())
            for ptm in ptms:
                if self.has_distractor_pair(term, ptm):
                    self.vocabulary_skipgram_matches[term].remove_phrase(ptm)
                elif self.vocabulary_skipgram_matches[term].match_type[ptm] is MatchType.NONE:
                    self.vocabulary_skipgram_matches[term].remove_phrase(ptm)

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
            if debug > 1:
                print('\n  find_skipgram_token_matches_in_text - text_token:', text_token)
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
            print(f'\n    find_skipgram_token_matches_for_token - text_token: {text_token}'
                  f'\tchar_index: {text_token.char_index}')
        if self.vocabulary.has_term(text_token):
            text_phrase_token_matches = get_vocabulary_skipgram_matches(text_token, self, debug=debug)
        else:
            text_phrase_token_matches = get_token_skipgram_matches(text_token, self, debug=debug)
        if debug > 1:
            print(f'    find_skipgram_token_matches_for_token - number of phrase matches: '
                  f'{len(text_phrase_token_matches.match_start_offsets)}')
            print()
        token_match_types = set()
        for pmi, phrase_token_match in enumerate(text_phrase_token_matches.matches):
            if debug > 2:
                print(f'\n    find_skipgram_token_matches_for_token - text_token: {text_token}'
                      f'\tchar_index: {text_token.char_index}')
                print(f'\n    find_skipgram_token_matches_for_token - phrase_token_match {pmi+1}: {phrase_token_match}')
            if debug > 2:
                print(f'    find_skipgram_token_matches_for_token - number of skipgram matches: '
                      f'{len(text_phrase_token_matches.match_start_offsets[phrase_token_match])}')
                print(f'    find_skipgram_token_matches_for_token - skipgram matches start offset: ',
                      text_phrase_token_matches.match_skipgrams[phrase_token_match][0].start_offset, '\tend_offset',
                      text_phrase_token_matches.match_skipgrams[phrase_token_match][-1].end_offset,
                      )
            # print(text_token.n, phrase_token_match, self.has_distractor_pair(text_token.n, phrase_token_match))
            # if self.has_distractor_pair(text_token.n, phrase_token_match):
                # skip this match if the text token is part of the vocabulary
                # and has been registered as a distractor for the phrase token
            #     continue
            match_type = text_phrase_token_matches.match_type[phrase_token_match]
            """
            match_type = get_token_skip_match_type(text_token.normalised_string,
                                                   text_token_num_skips,
                                                   token_skip_matches,
                                                   phrase_token_match, self, debug=debug)
            """
            if debug > 3:
                print(f'    find_skipgram_token_matches - match_type: {match_type}')
            token_match_types.add(match_type)
            if match_type == MatchType.NONE:
                continue
            elif match_type == MatchType.FULL:
                token_match = TokenMatch(text_token, phrase_token_match, match_type)
                if debug > 2:
                    print(f'\n    find_skipgram_token_matches - adding full match: {token_match}\n')
                token_matches.append(token_match)
                if debug > 2:
                    print('NUMBER OF TOKEN MATCHES:', len(token_matches))
            elif match_type == MatchType.PARTIAL_OF_PHRASE_TOKEN:
                if debug > 3:
                    print(f'\tfind_skipgram_token_matches - partial_matches: {partial_matches}')
                if phrase_token_match in partial_matches:
                    if debug > 3:
                        print(phrase_token_match)
                        print(partial_matches)
                        print(partial_matches[phrase_token_match])
                    last_partial = partial_matches[phrase_token_match][-1]
                    if debug > 3:
                        print(f'\tfind_skipgram_token_matches - last_partial: {last_partial}')
                    if text_token.char_index - (last_partial.char_index + len(last_partial)) > 4:
                        if debug > 3:
                            print(f'\t\tfind_skipgram_token_matches - text_token.char_index: {text_token.char_index}')
                            print(f'\t\tfind_skipgram_token_matches - end of last_partial: '
                                  f'{last_partial.char_index + len(last_partial)}')
                            print(f'\t\tfind_skipgram_token_matches - removing partial_match for phrase token: '
                                  f'{phrase_token_match}')
                        del partial_matches[phrase_token_match]
                if debug > 3:
                    print(f'\tfind_skipgram_token_matches - adding partial "{text_token}" '
                          f'of phrase token "{phrase_token_match}"')
                text_tuple = self.terms_to_tuple([t.n for t in partial_matches[phrase_token_match]] + [text_token.n])
                if self.has_distractor_pair(text_tuple, phrase_token_match):
                    if debug > 1:
                        print("current partial match is a distractor pair")
                    continue
                else:
                    partial_matches[phrase_token_match].append(text_token)
                if len(partial_matches[phrase_token_match]) > 1:
                    # print(f'text_tuple: {text_tuple}\tphrase_token_match: {phrase_token_match}')
                    if debug > 2:
                        print(f'\t\tfind_skipgram_token_matches - checking if partial_matches align with '
                              f'phrase token: {partial_matches[phrase_token_match]}')
                    first_partial = partial_matches[phrase_token_match][0]
                    last_partial = partial_matches[phrase_token_match][-1]
                    partial_length = last_partial.char_index + len(last_partial) - first_partial.char_index
                    length_diff = partial_length - len(phrase_token_match)
                    if debug > 2:
                        print(f'\t\tfind_skipgram_token_matches - '
                              f'partial_matches: {partial_matches[phrase_token_match]}')
                        print(f'\t\tfind_skipgram_token_matches - partial_length: {partial_length}')
                        print(f'\t\tfind_skipgram_token_matches - phrase_token length: {phrase_token_match}')
                        print(f'\t\tfind_skipgram_token_matches - lenght_diff: {length_diff}')

                    if length_diff > 0 or abs(length_diff) <= self.config['max_token_length_variance']:
                        token_match = TokenMatch(partial_matches[phrase_token_match],
                                                 phrase_token_match, match_type)
                        token_matches.append(token_match)
                        if debug > 1:
                            print('NUMBER OF TOKEN MATCHES:', len(token_matches))
                        if debug > 2:
                            print(f'\n    find_skipgram_token_matches - adding full match: {token_match}')
                        partial_matches[phrase_token_match].pop(0)
            elif match_type == MatchType.PARTIAL_OF_TEXT_TOKEN:
                token_match = TokenMatch(text_token, phrase_token_match, match_type)
                if debug > 1:
                    print('NUMBER OF TOKEN MATCHES:', len(token_matches))
                if debug > 1:
                    print(f'\n    find_skipgram_token_matches - adding partial of text token match: {token_match}\n')
                token_matches.append(token_match)
        if MatchType.PARTIAL_OF_PHRASE_TOKEN not in token_match_types:
            if debug > 2:
                print(f'    find_skipgram_token_matches - emptying partial_matches')
            partial_matches = defaultdict(list)
        if debug > 1:
            for phrase_token_match in text_phrase_token_matches.match_start_offsets:
                if phrase_token_match not in partial_matches:
                    continue
                if text_token in partial_matches[phrase_token_match]:
                    print(f"\t\tpartial_matches '{phrase_token_match}': {partial_matches[phrase_token_match]}\n")

    def _pick_best_candidates(self, doc: Doc, candidate_phrases: Dict[Phrase, List[PartialPhraseMatch]],
                              debug: int = 0):
        debug = debug + 1
        if debug > 2:
            print('\n  _pick_best_candidates - listing partial matches:')
            for phrase in candidate_phrases:
                print('    _pick_best_candidates - phrase:', phrase)
                for partial in candidate_phrases[phrase]:
                    print('\t_pick_best_candidates - partial:', partial)
        phrase_at_offset: Dict[int, PartialPhraseMatch] = {}
        for phrase in candidate_phrases:
            for pp in candidate_phrases[phrase]:
                full_text_length = pp.text_end - pp.text_start
                # the match_string is only the text tokens, not the text between
                # start and end offsets: if there are non-matching tokens in between,
                # the Levenshtein score should be computed only on the matching text
                # tokens.
                pp.match_string = ' '.join([token.n for token in pp.text_tokens])
                if debug > 1:
                    print('\n    _pick_best_candidates - pp:', pp.phrase.phrase_string)
                    print('\t_pick_best_candidates - pp.match_string:', pp.match_string)
                length_diff = abs(len(pp.match_string) - len(pp.phrase.phrase_string))
                if length_diff > self.max_length_variance:
                    if abs(full_text_length - len(pp.phrase.phrase_string)) > self.max_length_variance:
                        if debug > 1:
                            print(f"      length difference above max length variance")
                        continue
                    elif 1 - (length_diff / len(pp.phrase.phrase_string)) < self.config['levenshtein_threshold']:
                        if debug > 1:
                            print(f"      1 - (length difference over phrase length) is below levenshtein_threshold")
                        continue
                    else:
                        if debug > 1:
                            print(f"      length difference above max length variance, but full text length below")
                # distance_cutoff = len(pp.phrase.phrase_string) * (1 - self.config['levenshtein_threshold'])
                # pp.levenshtein_score = score_levenshtein_similarity_ratio(pp.phrase.phrase_string, pp.match_string)
                pp.levenshtein_score = score_ratio(pp.phrase.phrase_string, pp.match_string,
                                                   score_cutoff=self.config['levenshtein_threshold'])
                # pp.levenshtein_score = distance(pp.phrase.phrase_string, pp.match_string,
                #                                 score_cutoff=distance_cutoff)
                if debug > 1:
                    print('\t_pick_best_candidates - pp.levenshtein_score:', pp.levenshtein_score)
                if pp.levenshtein_score < self.config['levenshtein_threshold']:
                    if debug > 1:
                        print(f'\t\t_pick_best_candidates - phrase below '
                              f'score threshold {pp.text_start}:', pp.phrase.phrase_string)
                    continue
                if pp.text_start in phrase_at_offset:
                    if debug > 1:
                        print(f'\t\t_pick_best_candidates - next phrase at '
                              f'offset {pp.text_start}:', pp.phrase.phrase_string)
                    if pp.levenshtein_score > phrase_at_offset[pp.text_start].levenshtein_score:
                        if debug > 1:
                            print(f'\t\t_pick_best_candidates - replacing '
                                  f'phrase at offset {pp.text_start}:', pp.phrase.phrase_string)
                        phrase_at_offset[pp.text_start] = pp
                else:
                    if debug > 1:
                        print(f'\t\t_pick_best_candidates - first phrase at '
                              f'offset {pp.text_start}:', pp.phrase.phrase_string)
                    phrase_at_offset[pp.text_start] = pp
        phrases = []
        for partial_phrase in sorted(phrase_at_offset.values(), key=lambda x: x.text_start):
            if partial_phrase.phrase.phrase_string in self.phrase_model.phrase_index:
                phrase = partial_phrase.phrase
                variant = phrase
            elif partial_phrase.phrase.phrase_string in self.phrase_model.variant_index:
                phrase = self.phrase_model.variant_of(partial_phrase.phrase)
                variant = partial_phrase.phrase
            elif partial_phrase.phrase.phrase_string in self.phrase_model.distractor_index:
                # phrase is a distractor
                continue
            else:
                raise KeyError(f"partial phrase {partial_phrase.phrase} not registered in phrase_model.")
            phrase = PhraseMatch(phrase, variant, partial_phrase.match_string,
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
        step1, step2 = None, None
        doc = get_tokenized_doc(text, self.tokenizer)
        if debug > 0:
            print('find_matches - checking token matches in text')
        token_matches = self.find_skipgram_token_matches_in_text(doc, debug=debug)
        if debug > 0:
            print('find_matches - number of token_matches:', len(token_matches))
            step1 = time.time()
            print(f'step 1 took: {step1 - start: >.2f} seconds')
        if debug > 1:
            for tm in token_matches:
                print('find_matches - token_match:', tm)
        candidate_phrases = get_partial_phrases(token_matches, self, debug=debug)
        if debug > 0:
            print('find_matches - number of candidate phrases:', len(candidate_phrases))
            print('find_matches - number of partial phrases:',
                  sum([len(candidate_phrases[phrase]) for phrase in candidate_phrases]))
            step2 = time.time()
            print(f'step 2 took: {step2 - step1: >.2f} seconds')
        matches = self._pick_best_candidates(doc, candidate_phrases, debug=debug)
        filtered_matches = self.filter_matches_by_offset_threshold(matches)
        if debug > 0:
            print('find_matches - number of matches:', len(matches))
            print('find_filtered_matches - number of filtered_matches:', len(filtered_matches))
            step3 = time.time()
            print(f'step 3 took: {step3 - step2: >.2f} seconds')
        return filtered_matches


def is_distractor(text_token: str, phrase_token: str, dist_threshold: int = 2, debug: int = 0):
    """Check if a text token is a distractor for a phrase token."""
    dist = distance(text_token, phrase_token)
    length_diff = abs(len(text_token) - len(phrase_token))
    if dist - length_diff == 0 and len(text_token) <= 3:
        return text_token not in phrase_token
    if debug > 1:
        print(f"token_searcher.is_distractor - text_token '{text_token}', phrase_token: '{phrase_token}'")
        print(f"    dist: {dist}, length_diff: {length_diff}")
    # if len(text_token) >= 10 and 2 < dist - length_diff < 5:
    #     pass
    return dist - length_diff > dist_threshold


def token_is_out_of_phrase_range(token: Token, phrase: Phrase, token_searcher: FuzzyTokenSearcher):
    assert token.n in token_searcher.phrase_model.min_token_offset_in_phrase
    assert phrase.phrase_string in token_searcher.phrase_model.min_token_offset_in_phrase[token.n]
    if has_max_start_offset(phrase):
        max_token_offset = token_searcher.phrase_model.max_token_offset_in_phrase[token.n][phrase.phrase_string]
        if token.char_index > phrase.max_start_offset + max_token_offset:
            return True
    if has_max_end_offset(phrase):
        min_token_offset = token_searcher.phrase_model.min_token_offset_in_phrase[token.n][phrase.phrase_string]
        if token.char_index < phrase.max_end_offset + min_token_offset:
            return True
    return False


def get_partial_phrases(token_matches: List[TokenMatch], token_searcher: FuzzyTokenSearcher,
                        max_char_gap: int = 20, debug: int = 0):
    partial_phrase: Dict[Phrase, PartialPhraseMatch] = {}
    open_partials: Dict[Phrase, List[PartialPhraseMatch]] = defaultdict(list)
    candidate_phrase: Dict[Phrase, List[PartialPhraseMatch]] = defaultdict(list)
    prev_token_match = None
    max_partial_start_offset = 5
    if debug > 1:
        print(f"token_searcher.get_partial_phrases")
    for token_match in token_matches:
        if prev_token_match and token_match.text_start - prev_token_match.text_end > max_char_gap:
            if debug > 1:
                gap = token_match.text_start - prev_token_match.text_end
                print(f"    gap between prev_token_match and token_match ({gap}) "
                      f"bigger than max_char_gap {max_char_gap}")
                print(f"      emptying partial_phrase {partial_phrase}")
            partial_phrase: Dict[Phrase, PartialPhraseMatch] = {}
        if debug > 1:
            print("\n--------------\n"
                  f"text start: {token_match.text_start: >4}\tend{token_match.text_end: >4}\t\t"
                  f"text tokens: {token_match.text_tokens}\t\tphrase tokens{token_match.phrase_tokens}")
        for phrase_token_string in token_match.phrase_tokens:
            if debug > 1:
                print('\tphrase_token_string:', phrase_token_string)
            for phrase_string in token_searcher.phrase_model.token_in_phrase[phrase_token_string]:
                # phrase = token_searcher.phrase_model.phrase_index[phrase_string]
                phrase = token_searcher.phrase_model.get_phrase(phrase_string)
                # print('phrase:', phrase)
                # print('phrase.tokens:', phrase.tokens)
                if debug > 1 and phrase is not None:
                    print('\t\tphrase:', phrase)
                if isinstance(phrase, Phrase) is False:
                    continue
                if phrase not in open_partials:
                    partial = PartialPhraseMatch(phrase, [token_match])
                    offset = token_searcher.phrase_model.min_token_offset_in_phrase[phrase_token_string][
                        phrase.phrase_string]
                    # Idea 2024-12-07:
                    # do not add a new partial for a phrase, based on a token that is far from the start
                    # of that phrase
                    if offset < max_partial_start_offset:
                        if debug > 1:
                            print('\t\t\tADDING PHRASE AND APPENDING PARTIAL')
                            print(f'\t\t\t{offset}\t{partial.text_start}')
                        # partial_phrase[phrase] = partial
                        open_partials[phrase].append(partial)
                elif isinstance(phrase, Phrase):
                    added = False
                    partials = [partial for partial in open_partials[phrase]]
                    for partial in partials:
                        # check if token extends existing partial match
                        # check 1: if token is too far from partial end, move partial to candidates
                        if token_match.text_start - partial.text_end > max_char_gap:
                            candidate_phrase[phrase].append(partial)
                            open_partials[phrase].remove(partial)
                            if debug > 1:
                                offset = token_searcher.phrase_model.min_token_offset_in_phrase[phrase_token_string][
                                    phrase.phrase_string]
                                print('\t\t\tMOVING PARTIAL FROM OPEN TO CANDIDATE')
                                print(f'\t\t\t{offset}\t{partial.text_start}')
                            # print('skipping partial match:', token_match.text_start, partial.text_end)
                            continue
                        if debug > 2:
                            print(f"TEST partial.text_end: {partial.text_end} "
                                  f"token_match.text_start: {token_match.text_start}")
                            print(f"    partial.missing_tokens: {partial.missing_tokens}")
                            print(f"    token_match.phrase_tokens: {token_match.phrase_tokens}")
                        if partial.text_end < token_match.text_start and \
                                any([pt for pt in token_match.phrase_tokens if pt in partial.missing_tokens]):
                            partial_copy = copy_partial_match(partial)
                            partial_copy.add_tokens([token_match])
                            if partial_copy.text_length - len(phrase) <= token_searcher.config['max_length_variance']:
                                open_partials[phrase].append(partial_copy)
                                if debug > 1:
                                    print('\t\t\tADDING TOKEN TO COPY OF PARTIAL')
                                    print('\t\t', partial_copy.text_tokens)
                                    print('\t\t', partial_copy.text_start, partial_copy.text_end)
                                    print('\t\t', partial_copy.text_length, len(phrase))
                                added = True
                        # elif partial.text_end < token_match.text_start:
                    if not added:
                        offset = token_searcher.phrase_model.min_token_offset_in_phrase[phrase_token_string][
                            phrase.phrase_string]
                        if offset < max_partial_start_offset:
                            partial = PartialPhraseMatch(phrase, [token_match])
                            if debug > 1:
                                print('\t\t\tADDING AS NEW PARTIAL TO PHRASE')
                                print('\t\t', partial.text_tokens)
                                print('\t\t', partial.text_start, partial.text_end)
                                print('\t\t', partial.text_length, len(phrase))
                            open_partials[phrase].append(partial)
            if debug > 1:
                print(f"\nOPEN PARTIALS:")
                for phrase in open_partials:
                    print(f"\n    PHRASE: {phrase}")
                    for partial in open_partials[phrase]:
                        print(f"\tPARTIAL: {partial.text_tokens}, {partial.phrase_tokens}\t "
                              f"RANGE: {partial.text_start}-{partial.text_end}\t"
                              f"MISSING: {partial.missing_tokens}")
                print('---------------')
        prev_token_match = token_match
    for phrase in open_partials:
        for partial in open_partials[phrase]:
            candidate_phrase[phrase].append(partial)
    for phrase in candidate_phrase:
        remove_partials = []
        remove_incomplete = False
        if any([len(partial.missing_tokens) == 0 for partial in candidate_phrase[phrase]]):
            remove_incomplete = True
        for partial in candidate_phrase[phrase]:
            if debug > 2:
                print('\tCHECKING PARTIAL CANDIDATE')
                print('\t\t', partial.text_tokens)
                print('\t\t', partial.text_start, partial.text_end)
                print('\t\t', partial.text_length, len(phrase))
            if remove_incomplete is True and len(partial.missing_tokens) > 0:
                if debug > 2:
                    print("\t\tREMOVE CANDIDATE BECAUSE IT'S INCOMPLETE!")
                remove_partials.append(partial)
                # candidate_phrase[phrase].remove(partial)
            elif abs(partial.text_length - len(phrase)) > token_searcher.config['max_length_variance']:
                if debug > 2:
                    print('\t\tREMOVE CANDIDATE!')
                remove_partials.append(partial)
                # candidate_phrase[phrase].remove(partial)
                # candidate_phrase.append(partial)
                # del partial
            # elif partial.text_length > len(phrase) + token_searcher.config['max_length_variance']:
            else:
                if debug > 2:
                    print('\t\tCANDIDATE FOUND!')
                # del partial[phrase]
        for partial in remove_partials:
            candidate_phrase[phrase].remove(partial)
    return candidate_phrase


def token_within_phrase_offset(token_searcher: FuzzyTokenSearcher, text_token: Token,
                               phrase_token: str, debug: int = 0):
    if phrase_token in token_searcher.phrase_model.phrase_token_max_start_offset:
        max_token_offset = token_searcher.phrase_model.phrase_token_max_start_offset[phrase_token]
        if debug > 4:
            print(f"token_searcher.token_within_phrase_offset:")
            print(f"    phrase_token: {phrase_token}")
            print(f"    max_token_offset: {max_token_offset}")
        if text_token.char_index > max_token_offset:
            return False
    if phrase_token in token_searcher.phrase_model.phrase_token_max_end_offset:
        max_token_offset = token_searcher.phrase_model.phrase_token_max_end_offset[phrase_token]
        if debug > 4:
            print(f"token_searcher.token_within_phrase_offset:")
            print(f"    phrase_token: {phrase_token}")
            print(f"    max_token_offset: {max_token_offset}")
        if text_token.char_end_index > max_token_offset:
            return False
    return True


def get_vocabulary_skipgram_matches(text_token: Token, token_searcher: FuzzyTokenSearcher, debug: int = 0):
    vsm = token_searcher.vocabulary_skipgram_matches[text_token.n]
    tsm = SkipMatches(token_searcher.ngram_size, token_searcher.skip_size)
    for phrase_token in vsm.match_start_offsets:
        if vsm.match_type == MatchType.NONE:
            continue
        if not token_within_phrase_offset(token_searcher, text_token, phrase_token, debug=debug):
            continue
        tsm.matches.add(phrase_token)
        tsm.match_start_offsets[phrase_token] = [so + text_token.char_index
                                                 for so in vsm.match_start_offsets[phrase_token]]
        tsm.match_end_offsets[phrase_token] = [eo + text_token.char_index
                                               for eo in vsm.match_end_offsets[phrase_token]]
        tsm.match_type[phrase_token] = vsm.match_type[phrase_token]
        tsm.match_skipgrams[phrase_token] = [sg for sg in vsm.match_skipgrams[phrase_token]]
    return tsm


def get_token_skipgram_matches(text_token: Token,
                               token_searcher: FuzzyTokenSearcher,
                               debug: int = 0):
    text_token_skips = [skipgram for skipgram in token2skipgrams(text_token.n, token_searcher.ngram_size,
                                                                 token_searcher.skip_size,
                                                                 pad_token=token_searcher.config['pad_token'])]
    if debug > 3:
        print(f'\n    find_skipgram_token_matches_for_token - text_token skips: {text_token_skips}')
    token_skip_matches = SkipMatches(token_searcher.ngram_size, token_searcher.skip_size)
    for skipgram in text_token_skips:
        for phrase_token in token_searcher.token_skipgram_index[skipgram.string]:
            if debug > 4:
                print(f"token_searcher.get_token_skipgram_matches:")
                print(f"    phrase_token: {phrase_token}\ttype: {type(phrase_token)}")
                print(f"    in max_start_offset: "
                      f"{phrase_token in token_searcher.phrase_model.phrase_token_max_start_offset}")
            if token_searcher.has_distractor_pair((text_token.n,), (phrase_token, )):
                if debug > 4:
                    print(f"token_searcher.get_token_skipgram_matches:")
                    print(f"    phrase_token: {phrase_token}\ttype: {type(phrase_token)}")
                    print(f"    in distractor pairs.")
                continue
            if token_within_phrase_offset(token_searcher, text_token, phrase_token, debug=debug):
                token_skip_matches.add_skip_match(skipgram, phrase_token)
    get_token_skip_match_types(token_searcher, text_token, token_skip_matches, text_token_skips, debug=debug)
    return token_skip_matches


def get_token_skip_match_types(token_searcher: FuzzyTokenSearcher, text_token: Token,
                               token_skip_matches: SkipMatches, text_token_skips, debug: int = 0):
    for phrase_token_match in token_skip_matches.match_start_offsets:
        match_type = get_token_skip_match_type(text_token.normalised_string, len(text_token_skips),
                                               token_skip_matches, phrase_token_match, token_searcher,
                                               debug=debug)
        token_skip_matches.match_type[phrase_token_match] = match_type


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
    if debug > 2:
        print(f"        get_token_skip_match_type - first:", first)
        print(f"        get_token_skip_match_type - last:", last)
        print(f"        get_token_skip_match_type - overlap_start:", overlap_start)
        print(f"        get_token_skip_match_type - overlap_end:", overlap_end)
        print(f"        get_token_skip_match_type - num_skip_matches:", num_skip_matches)
        print(f"        get_token_skip_match_type - text_token_skip_overlap:", text_token_skip_overlap)
        print(f"        get_token_skip_match_type - phrase_token_skip_overlap:", phrase_token_skip_overlap)
        print(f"        get_token_skip_match_type - length_variance:", length_variance)
    if text_token_skip_overlap < token_searcher.config['skipgram_threshold'] and \
            phrase_token_skip_overlap < token_searcher.config['skipgram_threshold']:
        match_type = MatchType.NONE
        if debug > 2:
            print(f"        get_token_skip_match_type - below skipgram thresholds, match_type:", match_type)
    elif length_variance > token_searcher.config['max_token_length_variance']:
        match_type = MatchType.NONE
        if debug > 2:
            print(f"        get_token_skip_match_type - above max length variance, match_type:", match_type)
    elif abs(len(text_token_string) - len(phrase_token_match)) <= token_searcher.config['max_token_length_variance']:
        match_type = MatchType.FULL
        if debug > 2:
            print(f"        get_token_skip_match_type - text and phrase tokens equal length, match_type:", match_type)
    elif len(text_token_string) < len(phrase_token_match):
        match_type = MatchType.PARTIAL_OF_PHRASE_TOKEN
        if debug > 2:
            print(f"        get_token_skip_match_type - phrase token longer than text token, match_type:", match_type)
    else:
        match_type = MatchType.PARTIAL_OF_TEXT_TOKEN
        if debug > 2:
            print(f"        get_token_skip_match_type - text token longer than phrase token, match_type:", match_type)
    return match_type
