"""Bridge between FuzzyPhraseSearcher and the optional Rust extension ``fuzzy_search_rs``.

The extension runs the skipgram scan, candidate building, word-boundary adjustment, per-phrase
overlap filtering and scoring of ``find_matches``. Exact matching and the final filters stay in
Python, and results are identical to the Python backend.

The searcher's ``backend`` setting chooses: ``'auto'`` (default: Rust when installed and
supported, otherwise Python), ``'python'`` or ``'rust'`` (error if unavailable). Build the extension with
``maturin build --release`` in the ``rust/`` directory and install the wheel.
"""
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from fuzzy_search.match.candidate_match import Candidate
from fuzzy_search.match.phrase_match import PhraseMatch
from fuzzy_search.phrase.phrase import Phrase

try:
    import fuzzy_search_rs
except ImportError:  # the extension is optional
    fuzzy_search_rs = None

BACKENDS = ('python', 'rust', 'auto')


def rust_available() -> bool:
    return fuzzy_search_rs is not None


def unsupported_reason(searcher, phrases: List[Phrase] = None) -> Union[None, str]:
    """Return why the Rust backend cannot reproduce this searcher, or None if it can."""
    if fuzzy_search_rs is None:
        return 'the fuzzy_search_rs extension is not installed'
    if searcher.ngram_size > fuzzy_search_rs.MAX_NGRAM_SIZE:
        return f'ngram_size {searcher.ngram_size} is larger than {fuzzy_search_rs.MAX_NGRAM_SIZE}'
    for phrase in phrases if phrases is not None else _indexed_phrases(searcher):
        if phrase.max_end_offset > 0 and not hasattr(phrase, 'max_end_end'):
            # the Python backend raises AttributeError for these phrases; keep that behaviour
            return f'phrase {phrase.phrase_string!r} uses max_end_offset'
    return None


def get_matcher(searcher) -> Union[None, 'RustMatcher']:
    """Return the searcher's Rust matcher, or None when the Python backend should run.

    The matcher is cached on the searcher and rebuilt when its indexes or settings change."""
    backend = getattr(searcher, 'backend', 'auto')
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, not {backend!r}")
    if backend == 'python':
        return None
    key = (searcher.ngram_size, searcher.skip_size, searcher.max_length_variance, searcher.skipgram_threshold,
           searcher.ignorecase, len(searcher.skipgram_index), len(searcher.variant_skipgram_index),
           len(searcher.phrases), len(searcher.variants), id(searcher.phrase_model))
    cached = searcher.__dict__.get('_rust_matcher')
    if cached is None or cached[0] != key:
        phrases = _indexed_phrases(searcher) if fuzzy_search_rs is not None else []
        reason = unsupported_reason(searcher, phrases)
        cached = (key, reason, None if reason else RustMatcher(searcher, phrases))
        searcher._rust_matcher = cached
    _, reason, matcher = cached
    if reason and backend == 'rust':
        raise RuntimeError(f'Rust backend unavailable: {reason}')
    return matcher


def lower_text_for_rust(text_string: str) -> Union[None, str]:
    """Return text_string.lower() when slicing it equals lowercasing the same slice of the original
    for every slice, so that Rust can score ignorecase matches; otherwise None.

    That fails when lowercasing changes a character's length (e.g. U+0130) and for the Greek capital
    sigma, whose lowercase depends on the following character."""
    lower = text_string.lower()
    if len(lower) != len(text_string) or '\u03a3' in text_string:
        return None
    return lower


def _indexed_phrases(searcher) -> List[Phrase]:
    phrases = {}
    for index in (searcher.skipgram_index, searcher.variant_skipgram_index):
        for phrase_set in index.values():
            for phrase in phrase_set:
                phrases[id(phrase)] = phrase
    # Python builds candidates per phrase in phrase_string order; ids follow that order
    return sorted(phrases.values(), key=lambda p: p.phrase_string)


class RustMatcher:
    """A SkipgramMatcher built from a searcher's indexes, plus the conversion of its output."""

    def __init__(self, searcher, phrases: List[Phrase] = None):
        self.phrases = phrases if phrases is not None else _indexed_phrases(searcher)
        self.phrase_id = {id(phrase): pi for pi, phrase in enumerate(self.phrases)}
        self.ignorecase = searcher.ignorecase
        model = searcher.phrase_model
        self.group = [model.is_variant_of.get(p.phrase_string, p.phrase_string) if model else p.phrase_string
                      for p in self.phrases]
        self.ids_by_string = defaultdict(list)
        for pi, phrase in enumerate(self.phrases):
            self.ids_by_string[phrase.phrase_string].append(pi)
        self.variant_ids = {self.phrase_id[id(p)] for ps in searcher.variant_skipgram_index.values() for p in ps}
        self.word_in_phrase = model.word_in_phrase if model else defaultdict(set)
        # the main phrase a match is credited to (match_offsets.candidates_to_matches)
        self.main_phrase = [model.phrase_index[model.is_variant_of[p.phrase_string]]
                            if model and p.phrase_string in model.is_variant_of else p for p in self.phrases]
        self.matcher = fuzzy_search_rs.SkipgramMatcher(
            searcher.ngram_size, searcher.skip_size, searcher.max_length_variance, searcher.skipgram_threshold,
            [self._phrase_tuple(p) for p in self.phrases],
            self._index(searcher.skipgram_index), self._index(searcher.variant_skipgram_index),
            [p.phrase_string for p in self.phrases], self._group_ids(),
            [p.phrase_string == g for p, g in zip(self.phrases, self.group)],
            [p.phrase_string.lower() for p in self.phrases])

    def _group_ids(self) -> List[int]:
        ids = {}
        group_ids = [ids.setdefault(g, len(ids)) for g in self.group]
        self.group_names = list(ids)
        return group_ids

    def _phrase_tuple(self, phrase: Phrase) -> tuple:
        if self.ignorecase:
            index, freq = phrase.skipgram_index_lower, phrase.skipgram_freq_lower
            early, late = phrase.early_skipgram_index_lower, phrase.late_skipgram_index_lower
        else:
            index, freq = phrase.skipgram_index, phrase.skipgram_freq
            early, late = phrase.early_skipgram_index, phrase.late_skipgram_index
        return (len(phrase.phrase_string), len(phrase.skipgram_set), phrase.num_skipgrams, phrase.late_threshold,
                phrase.max_start_offset, phrase.max_start_end, phrase.max_end_offset,
                getattr(phrase, 'max_end_end', -1),
                [(s, skips[0].start_offset) for s, skips in index.items() if skips],
                list(freq.items()), list(early), list(late))

    def _index(self, index) -> List[Tuple[str, List[int]]]:
        return [(s, [self.phrase_id[id(p)] for p in phrases]) for s, phrases in index.items() if phrases]

    def known_words(self, known_word_start_offset: Dict[int, Dict[str, any]]) -> list:
        """Convert exact-match known words into (start, end, excluded main ids, excluded variant ids)."""
        known = []
        for start, word in known_word_start_offset.items():
            match_phrases = word['match_phrases']
            in_phrase = self.word_in_phrase[word['word']]
            excluded_main = [pi for s in match_phrases for pi in self.ids_by_string.get(s, [])]
            excluded_variant = [pi for pi in self.variant_ids
                                if self.phrases[pi].phrase_string not in in_phrase
                                or self.phrases[pi].phrase_string in match_phrases]
            known.append((start, word['end'], excluded_main, excluded_variant))
        return known

    def scan_text(self, text: Dict[str, any]) -> str:
        return text['text'].lower() if self.ignorecase else text['text']

    def to_phrase_candidates(self, raw: List[tuple], group_order: List[int],
                             text: Dict[str, any]) -> Dict[str, List[Candidate]]:
        """Turn Rust output into get_skipmatch_candidates' result: candidates per main phrase string."""
        phrase_candidates = defaultdict(list)
        for gi in group_order:
            phrase_candidates[self.group_names[gi]] = []
        text_string = text['text']
        for pi, start, end, overlap in raw:
            phrase_candidates[self.group[pi]].append(
                Candidate(self.phrases[pi], start, end, text_string[start:end], overlap))
        return phrase_candidates

    def phrase_candidates(self, text: Dict[str, any], include_variants: bool,
                          known_word_start_offset: Dict[int, Dict[str, any]]) -> Dict[str, List[Candidate]]:
        raw, group_order = self.matcher.candidates(self.scan_text(text),
                                                   self.known_words(known_word_start_offset or {}),
                                                   include_variants)
        return self.to_phrase_candidates(raw, group_order, text)

    def filtered_candidates(self, text: Dict[str, any], include_variants: bool,
                            known_word_start_offset: Dict[int, Dict[str, any]], use_word_boundaries: bool,
                            punctuation: str) -> List[Candidate]:
        """Candidates as returned by FuzzyPhraseSearcher.filter_phrase_candidates."""
        raw = self.matcher.filtered_candidates(text['text'], self.scan_text(text),
                                               self.known_words(known_word_start_offset or {}),
                                               include_variants, use_word_boundaries, punctuation)
        return self.to_candidates(raw, text)

    def to_candidates(self, raw: List[tuple], text: Dict[str, any]) -> List[Candidate]:
        text_string = text['text']
        phrases = self.phrases
        return [Candidate(phrases[pi], start, end, text_string[start:end], overlap)
                for pi, start, end, overlap in raw]

    def thresholds(self, searcher) -> List[Tuple[float, float, float]]:
        """Thresholds per phrase id as FuzzyPhraseSearcher.filter_matches_by_threshold applies them:
        the variant's overrides the main phrase's, which overrides the searcher's."""
        out = []
        for variant, phrase in zip(self.phrases, self.main_phrase):
            t = {'char_match_threshold': searcher.char_match_threshold,
                 'ngram_threshold': searcher.ngram_threshold,
                 'levenshtein_threshold': searcher.levenshtein_threshold,
                 **phrase.thresholds, **variant.thresholds}
            out.append((t['char_match_threshold'], t['ngram_threshold'], t['levenshtein_threshold']))
        return out

    def scored_matches(self, searcher, text: Dict[str, any], include_variants: bool,
                       known_word_start_offset: Dict[int, Dict[str, any]],
                       use_word_boundaries: bool) -> List[PhraseMatch]:
        """The matches of candidates_to_matches followed by filter_matches_by_threshold."""
        lower = None
        if searcher.ignorecase:
            lower = lower_text_for_rust(text['text'])
            if lower is None or lower != self.scan_text(text):
                # scores need Python's str.lower() of each match string, so they are computed in Python
                candidates = self.filtered_candidates(text, include_variants, known_word_start_offset,
                                                      use_word_boundaries, searcher.punctuation)
                return self._score_in_python(searcher, candidates, text)
        raw = self.matcher.scored_candidates(text['text'], self.known_words(known_word_start_offset or {}),
                                             include_variants, use_word_boundaries, searcher.punctuation, lower)
        return self.to_matches(raw, text, self.thresholds(searcher), ignorecase=searcher.ignorecase)

    def to_matches(self, raw: List[tuple], text: Dict[str, any], thresholds,
                   ignorecase: bool = False) -> List[PhraseMatch]:
        matches = []
        text_string, text_id = text['text'], text['id']
        for pi, start, end, overlap, char_match, ngram_match, lev in raw:
            char_t, ngram_t, lev_t = thresholds[pi]
            if char_match < char_t or ngram_match < ngram_t or lev < lev_t:
                continue
            match = PhraseMatch(self.main_phrase[pi], self.phrases[pi], text_string[start:end], start,
                                text_id=text_id, ignorecase=ignorecase)
            match.character_overlap = char_match
            match.ngram_overlap = ngram_match
            match.levenshtein_similarity = lev
            match.skipgram_overlap = overlap
            matches.append(match)
        return matches

    def find_matches_threaded(self, searcher, texts: list, threads: int = 0, block_size: int = 2000,
                              use_word_boundaries: bool = None, allow_overlapping_matches: bool = None,
                              include_variants: bool = None, filter_distractors: bool = None,
                              skip_exact_matching: bool = None, first_best: bool = False,
                              debug: int = 0) -> List[List[PhraseMatch]]:
        """``[searcher.find_matches(text, ...) for text in texts]`` with the Rust stages of each block
        of texts running on `threads` threads (0 or negative: one per CPU) without the GIL.

        Exact matching and the final filtering steps run in Python, one text at a time."""
        if use_word_boundaries is None:
            use_word_boundaries = searcher.use_word_boundaries
        variants = searcher.include_variants if include_variants is None else include_variants
        threads = 0 if threads is None or threads < 0 else threads
        thresholds = self.thresholds(searcher)
        results = []
        for block_start in range(0, len(texts), block_size):
            prepared = [searcher._prepare_text(text, use_word_boundaries, include_variants, skip_exact_matching)
                        for text in texts[block_start:block_start + block_size]]
            known = [self.known_words(kw or {}) for _, _, kw in prepared]
            lowers = [lower_text_for_rust(t['text']) if searcher.ignorecase else None for t, _, _ in prepared]
            in_rust = [not searcher.ignorecase or (lw is not None and lw == self.scan_text(t))
                       for lw, (t, _, _) in zip(lowers, prepared)]
            rust_items = [(t['text'], k, lw) for (t, _, _), k, lw, ok in zip(prepared, known, lowers, in_rust) if ok]
            rust_raw = iter(self.matcher.scored_candidates_batch(rust_items, variants, use_word_boundaries,
                                                                 searcher.punctuation, threads))
            scored = []
            for (t, _, _), k, ok in zip(prepared, known, in_rust):
                if ok:
                    scored.append(self.to_matches(next(rust_raw), t, thresholds, ignorecase=searcher.ignorecase))
                else:
                    raw = self.matcher.filtered_candidates(t['text'], self.scan_text(t), k, variants,
                                                           use_word_boundaries, searcher.punctuation)
                    scored.append(self._score_in_python(searcher, self.to_candidates(raw, t), t))
            for matches, (_, exact_matches, _) in zip(scored, prepared):
                results.append(searcher._finish_matches(matches, exact_matches, filter_distractors=filter_distractors,
                                                        allow_overlapping_matches=allow_overlapping_matches,
                                                        first_best=first_best))
        return results

    def _score_in_python(self, searcher, candidates: List[Candidate], text: Dict[str, any]) -> List[PhraseMatch]:
        matches = []
        for candidate in candidates:
            match = PhraseMatch(self.main_phrase[self.phrase_id[id(candidate.phrase)]], candidate.phrase,
                                candidate.match_string, candidate.match_start_offset, text_id=text['id'],
                                ignorecase=searcher.ignorecase)
            match.add_scores(skipgram_overlap=candidate.skipgram_overlap)
            matches.append(match)
        return searcher.filter_matches_by_threshold(matches)
