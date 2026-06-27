"""Byte Pair Encoding (BPE) subword tokenization utilities.

This module implements a simple BPE training algorithm operating on tokens
split into characters (plus a trailing word-boundary marker). It builds a
corpus of ``BPEToken`` objects, repeatedly finds the most frequent adjacent
symbol pair using a ``FrequencyTracker`` index, merges that pair into a new
symbol across all tokens, and updates the frequency index incrementally,
producing a learned subword vocabulary.
"""

import time
from collections import Counter
from collections import defaultdict
from typing import Dict, Generator, List, Set, Tuple, Union


class BPEToken:
    """A token represented as a mutable list of BPE symbols.

    Initially each symbol is a single character of the token, with a
    trailing space symbol marking the end of the token. As BPE merges are
    applied, adjacent symbols get combined into longer symbols.

    Attributes:
        token (str): The original token string.
        symbols (List[str]): The current list of symbols representing the token.
    """

    def __init__(self, token: str):
        """Initializes a BPEToken by splitting it into characters plus an end-of-word marker.

        Args:
            token (str): The token string to represent.
        """
        self.token = token
        self.symbols = [char for char in token] + [' ']

    def __repr__(self):
        """Returns a string representation of the BPEToken."""
        return f"{self.__class__.__name__}(token='{self.token}', symbols={self.symbols})"

    @property
    def symbol_pairs(self):
        """List[Tuple[str, str]]: All adjacent pairs of symbols in the token."""
        return [(self.symbols[i], self.symbols[i+1]) for i in range(len(self.symbols) - 1)]


class FrequencyTracker:
    """Tracks symbol-pair frequencies and supports fast lookup of the most frequent pair(s).

    Frequencies are organized into buckets keyed first by frequency, then by the
    combined character length of the symbol pair, so that the most frequent pair
    (optionally the shortest among ties) can be retrieved without scanning all pairs.

    Attributes:
        freq_buckets (Dict[int, Dict[int, Set[Tuple[str, str]]]]): Maps
            frequency -> combined symbol length -> set of symbol pairs with that
            frequency and length.
        symbol_pair_freq (Dict[Tuple[str, str], int]): Maps each symbol pair to its
            current frequency.
        max_freq (int): The highest frequency currently present in ``freq_buckets``.
    """

    def __init__(self):
        # freq_buckets[freq][length] = set of elements
        self.freq_buckets = defaultdict(lambda: defaultdict(set))
        self.symbol_pair_freq = defaultdict(int)
        self.max_freq = 0

    def update(self, symbol_pair: Tuple[str, str], count: int):
        """Adjusts the frequency of a symbol pair by ``count`` and updates the buckets.

        Removes the pair from its old frequency/length bucket (if any) and re-inserts
        it into the bucket for its new frequency, also maintaining ``max_freq``. If the
        new frequency is zero or below, the pair is removed entirely.

        Args:
            symbol_pair (Tuple[str, str]): The symbol pair to update.
            count (int): The (positive or negative) amount to add to the pair's frequency.
        """
        if count == 0 or not isinstance(symbol_pair, tuple):
            return

        old_freq = self.symbol_pair_freq.get(symbol_pair, 0)
        new_freq = old_freq + count
        length = sum(len(symbol) for symbol in symbol_pair)

        if old_freq > 0:
            self.freq_buckets[old_freq][length].discard(symbol_pair)
            if not self.freq_buckets[old_freq][length]:
                del self.freq_buckets[old_freq][length]
            if not self.freq_buckets[old_freq]:
                del self.freq_buckets[old_freq]
                if old_freq == self.max_freq:
                    self.max_freq = max(self.freq_buckets.keys(), default=0)

        if new_freq > 0:
            self.symbol_pair_freq[symbol_pair] = new_freq
            self.freq_buckets[new_freq][length].add(symbol_pair)
            self.max_freq = max(self.max_freq, new_freq)
        elif symbol_pair in self.symbol_pair_freq:
            del self.symbol_pair_freq[symbol_pair]  # Remove entirely if frequency drops to 0 or below

    def most_frequent(self, length=None):
        """Returns one symbol pair with the current maximum frequency.

        Args:
            length (int, optional): If given, restrict the search to symbol pairs whose
                combined character length equals this value.

        Returns:
            Optional[Tuple[Tuple[str, str], int]]: A tuple of (symbol_pair, frequency),
            or None if there is no pair (matching the length, if given).
        """
        if self.max_freq == 0:
            return None
        if length is not None:
            symbol_pairs = self.freq_buckets[self.max_freq].get(length, set())
            if symbol_pairs:
                return next(iter(symbol_pairs)), self.max_freq
            else:
                return None
        else:
            min_length = min(self.freq_buckets[self.max_freq].keys())
            length_bucket = self.freq_buckets[self.max_freq][min_length]
            if length_bucket:
                return next(iter(length_bucket)), self.max_freq
            else:
                return None

    def frequency_of(self, symbol_pair):
        """Returns the current frequency of a symbol pair, or 0 if it is not tracked."""
        return self.symbol_pair_freq.get(symbol_pair, 0)

    def all_with_max_frequency(self, length=None):
        """Returns all symbol pairs that currently have the maximum frequency.

        Args:
            length (int, optional): If given, restrict to symbol pairs whose combined
                character length equals this value.

        Returns:
            Set[Tuple[str, str]]: The set of symbol pairs at the maximum frequency.
        """
        if self.max_freq == 0:
            return set()
        if length is not None:
            return set(self.freq_buckets[self.max_freq].get(length, set()))
        else:
            result = set()
            for group in self.freq_buckets[self.max_freq].values():
                result.update(group)
            return result

    def most_frequent_shortest(self):
        """Return one symbol pair with the highest frequency and, among ties, the shortest
        combined string length.

        Returns:
            Optional[Tuple[Tuple[str, str], int, int]]: A tuple of (symbol_pair, frequency,
            length), or None if no symbol pairs are tracked.
        """
        if self.max_freq == 0:
            return None

        # Get the inner dict: {length: set of symbol_pairents}
        length_buckets = self.freq_buckets[self.max_freq]
        if not length_buckets:
            return None

        shortest_length = min(length_buckets.keys())
        symbol_pairs = length_buckets[shortest_length]
        if symbol_pairs:
            return next(iter(symbol_pairs)), self.max_freq, shortest_length
        return None


def string_tokens_to_corpus(tokens: List[str]) -> Counter[BPEToken, int]:
    """Builds a BPE corpus from a list of token strings.

    Args:
        tokens (List[str]): The token strings, e.g. words from a document collection.

    Returns:
        Counter[BPEToken, int]: A counter mapping each unique token (wrapped as a
        BPEToken) to its frequency in ``tokens``.
    """
    token_freq = Counter(tokens)
    corpus = Counter()
    for string_token in token_freq:
        bpe_token = BPEToken(string_token)
        corpus[bpe_token] = token_freq[string_token]
    return corpus


def generate_symbol_pairs(symbols: Union[List[str], Tuple[str, ...]]):
    """Returns all adjacent pairs of symbols in a sequence."""
    return [(symbols[i], symbols[i+1]) for i in range(len(symbols) - 1)]


def generate_corpus_symbol_pairs(corpus: Counter[BPEToken, int]) -> Generator[Tuple[Tuple[str, str], BPEToken], None, None]:
    """Iterate over all tokens in a corpus and return all tuples of two
    adjacent symbols together with their corresponding token"""
    for token in corpus:
        for symbol_pair in token.symbol_pairs:
            yield symbol_pair, token
    return None


def make_symbol_pair_freq(corpus: Counter[BPEToken, int],
                          symbol_pair_index: Dict[Tuple[str, str], Set[BPEToken]]):
    """Builds a FrequencyTracker of symbol-pair frequencies from a corpus and its symbol pair index.

    Args:
        corpus (Counter[BPEToken, int]): Token frequencies in the corpus.
        symbol_pair_index (Dict[Tuple[str, str], Set[BPEToken]]): Maps each symbol pair to
            the set of tokens it occurs in.

    Returns:
        FrequencyTracker: A tracker initialised with each symbol pair's total frequency,
        summed over the tokens it occurs in.
    """
    symbol_pair_freq = FrequencyTracker()
    for symbol_pair in symbol_pair_index:
        freq = sum([corpus[token] for token in symbol_pair_index[symbol_pair]])
        symbol_pair_freq.update(symbol_pair, freq)
    return symbol_pair_freq


def index_symbol_pair(corpus: Counter[BPEToken, int]) -> Dict[Tuple[str, str], Set[BPEToken]]:
    """Builds an index mapping each adjacent symbol pair to the set of tokens it occurs in.

    Args:
        corpus (Counter[BPEToken, int]): The tokens in the corpus.

    Returns:
        Dict[Tuple[str, str], Set[BPEToken]]: Maps each symbol pair to the tokens containing it.
    """
    symbol_pair_index = defaultdict(set)
    for symbol_pair, token in generate_corpus_symbol_pairs(corpus):
        symbol_pair_index[symbol_pair].add(token)
    return symbol_pair_index


def merge_symbols_in_token(merge_symbol: str, token: BPEToken):
    """Returns a new symbol sequence for a token with every occurrence of an adjacent
    symbol pair merged into a single combined symbol.

    Args:
        merge_symbol (str): The concatenated string of the symbol pair being merged
            (used to detect matching adjacent pairs in the token's symbol list).
        token (BPEToken): The token whose symbols are being merged.

    Returns:
        Tuple[str, ...]: The token's new symbol sequence, with matching adjacent pairs
        replaced by the merged symbol.
    """
    new_symbols = []
    skip = False
    for ti, symbol_pair in enumerate(token.symbol_pairs):
        if skip:
            skip = False
            continue
        symbol_pair = ''.join(symbol_pair)
        if symbol_pair == merge_symbol:
            skip = True
            new_symbols.append(symbol_pair)
        else:
            skip = False
            new_symbols.append(token.symbols[ti])
    if skip is False:
        new_symbols.append(token.symbols[-1])
    return tuple(new_symbols)


def find_new_symbol_pairs(merge_symbol: str, token: Tuple[str, ...]):
    """Finds the new adjacent symbol pairs introduced around each occurrence of a merged symbol.

    Args:
        merge_symbol (str): The newly merged symbol to search for in ``token``.
        token (Tuple[str, ...]): The token's (already merged) symbol sequence.

    Returns:
        List[Tuple[str, str]]: The new symbol pairs formed with ``merge_symbol``'s neighbours.
    """
    new_pairs = []
    for i in range(len(token)):
        if token[i] == merge_symbol:
            # print(f"i: {i}\ttoken[i]: {token[i]}")
            if i > 0:
                new_pair = token[i-1], token[i]
                new_pairs.append(new_pair)
            if i < len(token) - 1:
                new_pair = token[i], token[i+1]
                new_pairs.append(new_pair)
    return new_pairs


def compare_token_symbol_pairs(token1: Union[List[str], Tuple[str, ...]],
                                    token2: Union[List[str], Tuple[str, ...]]) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """Compares the adjacent symbol pairs of two symbol sequences (e.g. a token before
    and after a merge).

    Args:
        token1 (Union[List[str], Tuple[str, ...]]): The first symbol sequence.
        token2 (Union[List[str], Tuple[str, ...]]): The second symbol sequence.

    Returns:
        Tuple[Set, Set, Set]: A tuple of (pairs present in both, pairs only in
        ``token1``, pairs only in ``token2``).
    """
    pairs1 = set(generate_symbol_pairs(token1))
    pairs2 = set(generate_symbol_pairs(token2))
    overlap = pairs1.intersection(pairs2)
    only1 = pairs1 - pairs2
    only2 = pairs2 - pairs1
    return overlap, only1, only2


def merge_symbols_in_tokens(symbol_pair_index: Dict[Tuple[str, str], Set[BPEToken]],
                            symbol_pair_freq: FrequencyTracker,
                            corpus: Counter[BPEToken, int], merge_symbols: Tuple[str, str]):
    """Applies a BPE merge of ``merge_symbols`` to every token containing it, updating
    the symbol pair index and frequency tracker in place.

    For each affected token, this computes which symbol pairs disappear and which new
    ones appear as a result of the merge (via :func:`compare_token_symbol_pairs`), removes
    the disappearing pairs from the index/frequencies, adds the new ones, and finally
    removes the now-fully-merged ``merge_symbols`` entry from the index.

    Args:
        symbol_pair_index (Dict[Tuple[str, str], Set[BPEToken]]): The symbol pair to
            tokens index, updated in place.
        symbol_pair_freq (FrequencyTracker): The frequency tracker, updated in place.
        corpus (Counter[BPEToken, int]): Token frequencies in the corpus.
        merge_symbols (Tuple[str, str]): The symbol pair to merge.
    """
    merge_symbol = ''.join(merge_symbols)
    # print(f"corpus: {corpus}")
    update_tokens = [token for token in symbol_pair_index[merge_symbols]]
    for token in update_tokens:
        new_symbols = merge_symbols_in_token(merge_symbol, token)
        # print(f"token: {token}\tnew_token: {new_token}")
        # find the two symbols pairs that are only in the old tokens
        # and those only in the new token
        overlap, only1, only2 = compare_token_symbol_pairs(token.symbols, new_symbols)
        # print(f"overlap: {overlap}\tonly1: {only1}\tonly2: {only2}")
        # update the old two symbols index and frequency
        for old_symbol_pair in only1:
            symbol_pair_index[old_symbol_pair].remove(token)
            symbol_pair_freq.update(old_symbol_pair, -corpus[token])
            # print(f"updating symbol_pair_freq for old symbol_pair "
            #       f"{old_symbol_pair}: {symbol_pair_freq[old_symbol_pair]}")
        # update the new two symbols index and frequency
        for new_symbol_pair in only2:
            symbol_pair_index[new_symbol_pair].add(token)
            symbol_pair_freq.update(new_symbol_pair, corpus[token])
            # print(f"updating symbol_pair_freq for new symbol_pair "
            #       f"{new_symbol_pairs}: {symbol_pair_freq[new_symbol_pairs]}")
        # keep track of all old tokens so we can remove them from the corpus
        token.symbols = new_symbols
    # there should be no more tokens containing the two merge symbols,
    # so remove them from the two symbols index
    del symbol_pair_index[merge_symbols]


def generate_vocab(corpus: Counter[BPEToken, int]):
    """Collects the set of all distinct symbols currently used across a corpus of BPETokens.

    Args:
        corpus (Counter[BPEToken, int]): The tokens in the corpus.

    Returns:
        Set[str]: The set of distinct symbols in the corpus.
    """
    vocab = set()
    for token in corpus:
        for symbol in token.symbols:
            vocab.add(symbol)
    return vocab


def prune_symbol_pair_freq(symbol_pair_freq: Counter[Tuple[str, str], int]):
    """Removes symbol pairs with a frequency of zero from a symbol-pair frequency counter, in place."""
    remove_symbol_pair = [symbol_pair for symbol_pair in symbol_pair_freq if symbol_pair_freq[symbol_pair] == 0]
    num_freqs = len(set(symbol_pair_freq.values()))
    print(f"remove {len(remove_symbol_pair)} of {len(symbol_pair_freq)} symbol_pair, {num_freqs} distinct frequencies")
    for symbol_pair in remove_symbol_pair:
        del symbol_pair_freq[symbol_pair]


def make_byte_pair_encoding(tokens: List[str], k: int):
    """Trains a Byte Pair Encoding vocabulary from a list of token strings.

    Iteratively merges the most frequent (and, among ties, shortest) adjacent symbol
    pair across the corpus for ``k`` iterations, growing the vocabulary by one merged
    symbol per iteration.

    Args:
        tokens (List[str]): The token strings (e.g. words) to train the BPE model on.
        k (int): The number of merge iterations (and roughly the number of new
            subword symbols to learn).

    Returns:
        Set[str]: The learned vocabulary of symbols, including the original characters
        and the merged subword units.
    """
    corpus = string_tokens_to_corpus(tokens)
    symbol_pair_index = index_symbol_pair(corpus)
    symbol_pair_freq = make_symbol_pair_freq(corpus, symbol_pair_index)
    vocab = generate_vocab(corpus)
    # print(f"vocab: {vocab}")
    start = time.time()
    step_start = time.time()
    for iteration in range(k):
        # print("\n")
        # print(f"corpus: {corpus}")
        # print(f"symbol_pair_index: ")
        # for symbol_pair in symbol_pair_index:
        #     print(f"    {symbol_pair}:\t{symbol_pair_index[symbol_pair]}")
        merge_symbols, freq, length = symbol_pair_freq.most_frequent_shortest()
        # print(f"symbol_pair_freq: ")
        # for symbol_pair in symbol_pair_freq.freq_buckets[freq][length]:
        #     print(f"    {symbol_pair}:\tfreq: {freq}\tlength: {length}")

        # print(f"merge_symbols: {merge_symbols}")
        merge_symbols_in_tokens(symbol_pair_index, symbol_pair_freq, corpus, merge_symbols)
        merge_symbol = ''.join(merge_symbols)
        vocab.add(merge_symbol)
        if (iteration+1) % 100 == 0:
            # prune_symbol_pair_freq(symbol_pair_freq)
            step_end = time.time()
            took = step_end - step_start
            total = step_end - start
            print(f"iteration {iteration + 1} took {took:.5f} seconds, total {total:.2f} seconds - "
                  f"merge_symbols: {merge_symbols} ({merge_symbol})")
            step_start = time.time()
    return vocab
