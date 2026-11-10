"""The Rust backend (fuzzy_search_rs) must return exactly what the Python backend returns."""
import itertools
import pickle
from unittest import TestCase, skipIf

from fuzzy_search.search import rust_backend
from fuzzy_search.search.phrase_searcher import FuzzyPhraseSearcher

PHRASES = [
    {'phrase': 'Ontfangen een Missive van', 'variants': ['Ontfangen twee Missiven van']},
    {'phrase': 'is goedgevonden ende verstaan', 'levenshtein_threshold': 0.8},
    {'phrase': 'Den Heere', 'max_start_offset': 40},
    {'phrase': 'PRAESIDE,', 'label': 'president'},
    {'phrase': 'geinsereert.'},
    {'phrase': 'Op de Requeste van', 'variants': ['Op de Memorie van']},
]

TEXTS = [
    'Ontfangen een Missive vande Heere van Essen, is goetgevonden ende verstaen.',
    'Ontfangen twee missiven van den Heer Raadpensionaris. PRÆSIDE, Den Heere van Wassenaer.',
    'Den Heere komt hier te laat, want deze tekst is langer dan veertig tekens: Den Heere.',
    'IS goedtgevonden en verstaan dat Den Heere zal schrijven; is goedgevonden ende verstaan.',
    'Opde reqte. van Jan Pietersz, Op de memorie vanden Agent, op de Requeste  van',
    'gedelibereert, geinsereert. geinsereert geconsidereert,',
    '', 'a', 'Den', 'PRAESIDE,', ' Den Heere.',
    'İstanbul: DEN HEERE ende ΣΙΣΥΦΟΣ, Ontfangen een Missive van',  # lowercasing changes length / context
]


def as_tuples(matches):
    return [(m.phrase.phrase_string, m.variant.phrase_string, m.string, m.offset, m.end, m.label,
             m.character_overlap, m.ngram_overlap, m.levenshtein_similarity, m.skipgram_overlap)
            for m in matches]


def outcome(searcher, text, **kwargs):
    try:
        return as_tuples(searcher.find_matches(text, **kwargs))
    except Exception as err:  # both backends must fail the same way
        return type(err).__name__


@skipIf(not rust_backend.rust_available(), 'fuzzy_search_rs is not installed')
class TestRustBackendEqualsPython(TestCase):

    def pair(self, **config):
        python = FuzzyPhraseSearcher(phrase_model=PHRASES, config={**config, 'backend': 'python'})
        rust = FuzzyPhraseSearcher(phrase_model=PHRASES, config={**config, 'backend': 'rust'})
        return python, rust

    def test_settings_grid(self):
        for ngram_size, skip_size, ignorecase in itertools.product((1, 2, 3), (0, 1, 2), (False, True)):
            python, rust = self.pair(ngram_size=ngram_size, skip_size=skip_size, ignorecase=ignorecase,
                                     include_variants=True)
            for word_boundaries, variants, skip_exact in itertools.product((True, False), repeat=3):
                kwargs = dict(use_word_boundaries=word_boundaries, include_variants=variants,
                              skip_exact_matching=skip_exact)
                for text in TEXTS:
                    with self.subTest(n=ngram_size, k=skip_size, ignorecase=ignorecase, text=text, **kwargs):
                        self.assertEqual(outcome(python, text, **kwargs), outcome(rust, text, **kwargs))

    def test_overlap_and_distractor_options(self):
        python, rust = self.pair(include_variants=True, filter_distractors=True)
        for kwargs in (dict(allow_overlapping_matches=True), dict(first_best=True), {}):
            for text in TEXTS:
                self.assertEqual(outcome(python, text, **kwargs), outcome(rust, text, **kwargs))

    def test_threaded_batch_equals_serial_python(self):
        for ignorecase in (False, True):
            python, rust = self.pair(include_variants=True, ignorecase=ignorecase)
            texts = TEXTS * 5
            expected = [as_tuples(python.find_matches(text)) for text in texts]
            for workers in (None, 2, 0):
                result = rust.find_matches_batch(texts, workers=workers)
                self.assertEqual(expected, [as_tuples(ms) for ms in result])

    def test_threaded_batch_small_blocks(self):
        python, rust = self.pair(include_variants=True)
        matcher = rust_backend.get_matcher(rust)
        result = matcher.find_matches_threaded(rust, TEXTS, threads=2, block_size=3, first_best=True)
        self.assertEqual([as_tuples(python.find_matches(t, first_best=True)) for t in TEXTS],
                         [as_tuples(ms) for ms in result])

    def test_matches_use_searcher_phrases(self):
        _, rust = self.pair(include_variants=True)
        matches = rust.find_matches(TEXTS[1])
        self.assertTrue(matches)
        for match in matches:
            self.assertIs(match.phrase, rust.phrase_model.get_phrase(match.phrase.phrase_string))
            self.assertIs(match.variant, rust.phrase_model.get_phrase(match.variant.phrase_string))


@skipIf(not rust_backend.rust_available(), 'fuzzy_search_rs is not installed')
class TestRustBackendSelection(TestCase):

    def test_default_backend_is_auto(self):
        searcher = FuzzyPhraseSearcher(phrase_model=PHRASES)
        self.assertEqual(searcher.backend, 'auto')
        self.assertIsNotNone(rust_backend.get_matcher(searcher))

    def test_python_backend(self):
        searcher = FuzzyPhraseSearcher(phrase_model=PHRASES, config={'backend': 'python'})
        self.assertIsNone(rust_backend.get_matcher(searcher))

    def test_auto_uses_rust(self):
        searcher = FuzzyPhraseSearcher(phrase_model=PHRASES, config={'backend': 'auto'})
        self.assertIsNotNone(rust_backend.get_matcher(searcher))

    def test_invalid_backend(self):
        searcher = FuzzyPhraseSearcher(phrase_model=PHRASES, config={'backend': 'fortran'})
        with self.assertRaises(ValueError):
            searcher.find_matches(TEXTS[0])

    def test_matcher_rebuilt_after_config_change(self):
        searcher = FuzzyPhraseSearcher(phrase_model=PHRASES, config={'backend': 'rust'})
        first = rust_backend.get_matcher(searcher)
        self.assertIs(first, rust_backend.get_matcher(searcher))
        searcher.skipgram_threshold = 0.5
        self.assertIsNot(first, rust_backend.get_matcher(searcher))

    def test_large_ngram_size_falls_back(self):
        searcher = FuzzyPhraseSearcher(phrase_model=PHRASES, config={'backend': 'auto', 'ngram_size': 7})
        self.assertIsNone(rust_backend.get_matcher(searcher))
        strict = FuzzyPhraseSearcher(phrase_model=PHRASES, config={'backend': 'rust', 'ngram_size': 7})
        with self.assertRaises(RuntimeError):
            strict.find_matches(TEXTS[0])

    def test_pickle_drops_matcher(self):
        searcher = FuzzyPhraseSearcher(phrase_model=PHRASES, config={'backend': 'rust'})
        expected = as_tuples(searcher.find_matches(TEXTS[0]))
        copy = pickle.loads(pickle.dumps(searcher))
        self.assertNotIn('_rust_matcher', copy.__dict__)
        self.assertEqual(expected, as_tuples(copy.find_matches(TEXTS[0])))


class TestRustBackendUnavailable(TestCase):

    def test_auto_falls_back_and_rust_raises(self):
        saved = rust_backend.fuzzy_search_rs
        rust_backend.fuzzy_search_rs = None
        try:
            auto = FuzzyPhraseSearcher(phrase_model=PHRASES, config={'backend': 'auto'})
            self.assertIsNone(rust_backend.get_matcher(auto))
            strict = FuzzyPhraseSearcher(phrase_model=PHRASES, config={'backend': 'rust'})
            with self.assertRaises(RuntimeError):
                strict.find_matches(TEXTS[0])
        finally:
            rust_backend.fuzzy_search_rs = saved
