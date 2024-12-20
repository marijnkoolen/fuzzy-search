from typing import List, Generator, Tuple
from itertools import combinations

from Levenshtein import distance as score_distance
from Levenshtein import ratio as score_ratio


#################################
# String manipulation functions #
#################################

def make_ngrams(text: str, n: int) -> List[str]:
    """Turn a term string into a list of ngrams of size n

    :param text: a text string
    :type text: str
    :param n: the ngram size
    :type n: int
    :return: a list of ngrams
    :rtype: List[str]"""
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    if not isinstance(n, int):
        raise TypeError('n must be a positive integer')
    if n < 1:
        raise ValueError('n must be a positive integer')
    if n > len(text):
        return []
    text = "#{t}#".format(t=text)
    max_start = len(text) - n + 1
    return [text[start:start + n] for start in range(0, max_start)]


#####################################
# Term similarity scoring functions #
#####################################


def score_ngram_overlap(term1: str, term2: str, ngram_size: int):
    """Score the number of overlapping ngrams between two terms

    :param term1: a first term string
    :type term1: str
    :param term2: a second term string
    :type term2: str
    :param ngram_size: the character ngram size
    :type ngram_size: int
    :return: the number of overlapping ngrams
    :rtype: int
    """
    term1_ngrams = make_ngrams(term1, ngram_size)
    term2_ngrams = make_ngrams(term2, ngram_size)
    overlap = 0
    for ngram in term1_ngrams:
        if ngram in term2_ngrams:
            term2_ngrams.pop(term2_ngrams.index(ngram))
            overlap += 1
    return overlap


def score_ngram_overlap_ratio(term1, term2, ngram_size):
    """Score the number of overlapping ngrams between two terms as proportion of the length
    of the first term

    :param term1: a term string
    :type term1: str
    :param term2: a term string
    :type term2: str
    :param ngram_size: the character ngram size
    :type ngram_size: int
    :return: the number of overlapping ngrams
    :rtype: int
    """
    max_overlap = len(make_ngrams(term1, ngram_size))
    overlap = score_ngram_overlap(term1, term2, ngram_size)
    return overlap / max_overlap


def score_char_overlap_ratio(term1, term2):
    """Score the number of overlapping characters between two terms as proportion of the length
    of the first term

    :param term1: a term string
    :type term1: str
    :param term2: a term string
    :type term2: str
    :return: the number of overlapping ngrams
    :rtype: int
    """
    max_overlap = len(term1)
    overlap = score_char_overlap(term1, term2)
    return overlap / max_overlap


def score_char_overlap(term1: str, term2: str) -> int:
    """Count the number of overlapping character tokens in two strings.

    :param term1: a term string
    :type term1: str
    :param term2: a term string
    :type term2: str
    :return: the number of overlapping ngrams
    :rtype: int
    """
    num_char_matches = 0
    for char in term2:
        if char in term1:
            term1 = term1.replace(char, "", 1)
            num_char_matches += 1
    return num_char_matches


def score_levenshtein_similarity_ratio(term1, term2, score_cutoff: int = None):
    """Score the levenshtein similarity between two terms

    :param term1: a term string
    :type term1: str
    :param term2: a term string
    :type term2: str
    :param score_cutoff: the maximum distance beyond which distance calculation should be cut off
    :type score_cutoff: int
    :return: the number of overlapping ngrams
    :rtype: int
    """
    # max_distance = max(len(term1), len(term2))
    # distance = score_levenshtein_distance(term1, term2)
    # distance = score_distance(term1, term2, score_cutoff=score_cutoff)
    return score_ratio(term1, term2)
    # return 1 - distance / max_distance


def score_levenshtein_distance(term1: str, term2: str) -> int:
    """Calculate Levenshtein distance between two string.

    :param term1: a term string
    :type term1: str
    :param term2: a term string
    :type term2: str
    :return: the number of overlapping ngrams
    :rtype: int
    """
    if len(term1) > len(term2):
        term1, term2 = term2, term1
    distances = range(len(term1) + 1)
    for i2, c2 in enumerate(term2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(term1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


class SkipGram:

    def __init__(self, skipgram_string: str, start_offset: int, end_offset: int, skipgram_length: int):
        self.string = skipgram_string
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.length = skipgram_length

    def __repr__(self):
        return (f"{self.__class__.__name__}(string='{self.string}', start_offset={self.start_offset}, "
                f"end_offset={self.end_offset}, length={self.length})")


def insert_skips(window: str, skipgram_combinations: List[Tuple[int]]):
    """For a given skip gram window, return all skip grams for a given configuration."""
    for combination in skipgram_combinations:
        skip_gram = window[0]
        try:
            for index in combination:
                skip_gram += window[index]
            yield skip_gram, combination[-1] + 1, (0,) + combination
        except IndexError:
            pass


def text2skipgrams(text: str, ngram_size: int = 2, skip_size: int = 2) -> Generator[SkipGram, None, None]:
    """Turn a text string into a list of skipgrams.

    :param text: an text string
    :type text: str
    :param ngram_size: an integer indicating the number of characters in the ngram
    :type ngram_size: int
    :param skip_size: an integer indicating how many skip characters in the ngrams
    :type skip_size: int
    :return: An iterator returning tuples of skip_gram and offset
    :rtype: Generator[tuple]"""
    if ngram_size <= 0 or skip_size < 0:
        raise ValueError('ngram_size must be a positive integer, skip_size must be a positive integer or zero')
    if ngram_size == 1:
        for ci, char in enumerate(text):
            yield SkipGram(skipgram_string=char, start_offset=0,
                           end_offset=len(text) - ci + 1, skipgram_length=1)
        return None
    elif len(text) <= ngram_size:
        yield SkipGram(skipgram_string=text, start_offset=0, end_offset=0, skipgram_length=len(text))
        return None
    indexes = [i for i in range(0, ngram_size+skip_size)]
    skipgram_combinations = [combination for combination in combinations(indexes[1:], ngram_size-1)]
    for start_offset in range(0, len(text)-1):
        end_offset = len(text) - start_offset + 1
        window = text[start_offset:start_offset+ngram_size+skip_size]
        for skipgram, skipgram_length, combination in insert_skips(window, skipgram_combinations):
            yield SkipGram(skipgram_string=skipgram, start_offset=start_offset,
                           end_offset=end_offset, skipgram_length=skipgram_length)


def token2skipgrams(token: str, ngram_size: int = 2, skip_size: int = 2,
                    pad_token: bool = True) -> Generator[SkipGram, None, None]:
    """Turn a (padded) token string into a list of skipgrams.

    :param token: a token string
    :type token: str
    :param ngram_size: an integer indicating the number of characters in the ngram
    :type ngram_size: int
    :param skip_size: an integer indicating how many skip characters in the ngrams
    :type skip_size: int
    :param pad_token: a boolean flag to indicate whether padding should be included
                     at the boundaries of the token
    :type pad_token: bool
    :return: An iterator returning tuples of skip_gram and offset
    :rtype: Generator[tuple]"""
    token_length = len(token)
    if ngram_size <= 0 or skip_size < 0:
        raise ValueError('ngram_size must be a positive integer, skip_size must be a positive integer or zero')
    if pad_token is True:
        pad_size = ngram_size - 1
        padded_token = '#' * pad_size + token + '#' * pad_size
    else:
        pad_size = 0
        padded_token = token
    if ngram_size == 1:
        for ci, char in enumerate(token):
            yield SkipGram(skipgram_string=char, start_offset=0,
                           end_offset=len(token) - ci + 1, skipgram_length=1)
    elif token_length <= ngram_size and pad_token is False:
        yield SkipGram(skipgram_string=token, start_offset=0,
                       end_offset=0, skipgram_length=token_length)
    else:
        indexes = [i for i in range(0, ngram_size + skip_size)]
        skipgram_combinations = [combination for combination in combinations(indexes[1:], ngram_size - 1)]
        # print(skipgram_combinations)
        for start_offset in range(0, len(padded_token)):
            padded_start = start_offset
            window = padded_token[start_offset:start_offset+ngram_size+skip_size]
            # start_offset = 0 if start_offset < pad_size else start_offset - pad_size
            end_offset = token_length - start_offset + 1
            # print()
            # print(f'padded_token: {padded_token}')
            # print(f"start_offset: {start_offset}")
            # print(f"end_offset: {end_offset}")
            # print(f"window: {window}")
            if end_offset > token_length:
                end_offset = token_length
            for skipgram, skipgram_length, combination in insert_skips(window, skipgram_combinations):
                # print(f"combination: {combination}")
                combination = [idx+padded_start for idx in combination if pad_size <= idx+padded_start < token_length+pad_size]
                # print(f"combination: {combination}")
                if len(combination) == 0:
                    continue
                skipgram_length = combination[-1] - combination[0] + 1
                start_offset = combination[0] - pad_size
                # print(f"padded_start: {padded_start}\tstart_offset: {start_offset}\t"
                #       f"combintation: {combination}\tskipgram: {skipgram}\tskipgram_length: {skipgram_length}")
                # if pad_token is True and all([char == '#' for char in skipgram]):
                #     continue
                # if start_offset + skipgram_length > token_length:
                #     skipgram_length = token_length - start_offset
                yield SkipGram(skipgram_string=skipgram, start_offset=start_offset,
                               end_offset=end_offset, skipgram_length=skipgram_length)


non_word_affixes_2 = {
    ". ", ", ", "! ", "? ",
    " (", ") ", ").", ")!", "),", ")?",
    " [", "] ", "].", "]!", "],", "]?",
}
non_word_affixes_1 = {
    " ", ".", ",", "!", "?",
}


def get_non_word_prefix(string: str) -> str:
    """Check if a string has a non-word prefix and return it.

    :param string: the string from which the prefix is to be return
    :type string: str
    :return: the non-word prefix
    :rtype: str
    """
    if string[:2] in non_word_affixes_2:
        return string[:2]
    elif string[:1] in non_word_affixes_1:
        return string[:1]
    else:
        return ""


def get_non_word_suffix(string: str) -> str:
    """Check if a string has a non-word suffix and return it.

    :param string: the string from which the suffix is to be return
    :type string: str
    :return: the non-word suffix
    :rtype: str
    """
    if string[-2:] in non_word_affixes_2:
        return string[-2:]
    elif string[-1:] in non_word_affixes_1:
        return string[-1:]
    else:
        return ""


def strip_prefix(string: str) -> str:
    """Strip non-word prefix from string ending.

    :param string: the string from which the prefix is to be stripped
    :type string: str
    :return: the stripped string
    :rtype: str
    """
    if len(string) <= 2:
        pass
    elif string[:2] in non_word_affixes_2:
        string = string[2:]
    elif string[1] in [" ", ","]:
        string = string[2:]
    elif string[0] in [" ", ","]:
        string = string[1:]
    return string


def strip_suffix(string: str) -> str:
    """Strip non-word suffix from string ending.

    :param string: the string from which the suffix is to be stripped
    :type string: str
    :return: the stripped string
    :rtype: str
    """
    if len(string) <= 2:
        pass
    elif string[-2] in [" ", ","]:
        string = string[:-2]
    elif string[-2:] in [", ", ". ", "! ", "? "]:
        string = string[:-2]
    elif string[-1] in [" ", ","]:
        string = string[:-1]
    return string
