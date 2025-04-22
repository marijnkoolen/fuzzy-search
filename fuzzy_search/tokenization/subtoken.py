import time
from collections import Counter
from collections import defaultdict
from typing import Dict, Generator, List, Set, Tuple, Union


class BPEToken:

    def __init__(self, token: str):
        self.token = token
        self.symbols = [char for char in token] + [' ']

    def __repr__(self):
        return f"{self.__class__.__name__}(token='{self.token}', symbols={self.symbols})"

    @property
    def symbol_pairs(self):
        return [(self.symbols[i], self.symbols[i+1]) for i in range(len(self.symbols) - 1)]


class FrequencyTracker:

    def __init__(self):
        # freq_buckets[freq][length] = set of elements
        self.freq_buckets = defaultdict(lambda: defaultdict(set))
        self.symbol_pair_freq = defaultdict(int)
        self.max_freq = 0

    def update(self, symbol_pair: Tuple[str, str], count: int):
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
        return self.symbol_pair_freq.get(symbol_pair, 0)

    def all_with_max_frequency(self, length=None):
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
        """Return one symbol_pairent with the highest frequency and shortest string length."""
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
    token_freq = Counter(tokens)
    corpus = Counter()
    for string_token in token_freq:
        bpe_token = BPEToken(string_token)
        corpus[bpe_token] = token_freq[string_token]
    return corpus


def generate_symbol_pairs(symbols: Union[List[str], Tuple[str, ...]]):
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
    symbol_pair_freq = FrequencyTracker()
    for symbol_pair in symbol_pair_index:
        freq = sum([corpus[token] for token in symbol_pair_index[symbol_pair]])
        symbol_pair_freq.update(symbol_pair, freq)
    return symbol_pair_freq


def index_symbol_pair(corpus: Counter[BPEToken, int]) -> Dict[Tuple[str, str], Set[BPEToken]]:
    symbol_pair_index = defaultdict(set)
    for symbol_pair, token in generate_corpus_symbol_pairs(corpus):
        symbol_pair_index[symbol_pair].add(token)
    return symbol_pair_index


def merge_symbols_in_token(merge_symbol: str, token: BPEToken):
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
    pairs1 = set(generate_symbol_pairs(token1))
    pairs2 = set(generate_symbol_pairs(token2))
    overlap = pairs1.intersection(pairs2)
    only1 = pairs1 - pairs2
    only2 = pairs2 - pairs1
    return overlap, only1, only2


def merge_symbols_in_tokens(symbol_pair_index: Dict[Tuple[str, str], Set[BPEToken]],
                            symbol_pair_freq: FrequencyTracker,
                            corpus: Counter[BPEToken, int], merge_symbols: Tuple[str, str]):
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
    vocab = set()
    for token in corpus:
        for symbol in token.symbols:
            vocab.add(symbol)
    return vocab


def prune_symbol_pair_freq(symbol_pair_freq: Counter[Tuple[str, str], int]):
    remove_symbol_pair = [symbol_pair for symbol_pair in symbol_pair_freq if symbol_pair_freq[symbol_pair] == 0]
    num_freqs = len(set(symbol_pair_freq.values()))
    print(f"remove {len(remove_symbol_pair)} of {len(symbol_pair_freq)} symbol_pair, {num_freqs} distinct frequencies")
    for symbol_pair in remove_symbol_pair:
        del symbol_pair_freq[symbol_pair]


def make_byte_pair_encoding(tokens: List[str], k: int):
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
