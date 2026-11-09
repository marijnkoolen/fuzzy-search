from unittest import TestCase

from fuzzy_search.search.phrase_searcher import FuzzyPhraseSearcher


def match_tuples(matches):
    return [(m.phrase.phrase_string, m.variant.phrase_string, m.string, m.offset, m.levenshtein_similarity)
            for m in matches]


class TestFindMatchesBatch(TestCase):

    def setUp(self) -> None:
        phrases = [{'phrase': 'Ontfangen een Missive van', 'variants': ['Ontfangen twee Missiven van']},
                   {'phrase': 'is goedgevonden ende verstaan'},
                   {'phrase': 'Den Heere'}]
        self.searcher = FuzzyPhraseSearcher(phrase_model=phrases,
                                            config={'ngram_size': 2, 'skip_size': 2, 'include_variants': True})
        base = ['Ontfangen een Missive vande Heere van Essen, is goetgevonden ende verstaen.',
                'Ontfangen twee missiven van den Heer Raadpensionaris.',
                'Geen overeenkomst in deze tekst.',
                'IS goedtgevonden en verstaan dat Den Heere zal schrijven.']
        self.texts = [text + f' {i}' for i in range(15) for text in base]

    def test_serial_equals_find_matches(self):
        expected = [match_tuples(self.searcher.find_matches(t)) for t in self.texts]
        actual = [match_tuples(ms) for ms in self.searcher.find_matches_batch(self.texts)]
        self.assertEqual(expected, actual)

    def test_parallel_equals_serial(self):
        expected = [match_tuples(ms) for ms in self.searcher.find_matches_batch(self.texts)]
        result = self.searcher.find_matches_batch(self.texts, workers=2, chunk_size=7)
        self.assertEqual(expected, [match_tuples(ms) for ms in result])
        self.assertTrue(any(result))

    def test_parallel_matches_use_searcher_phrases(self):
        result = self.searcher.find_matches_batch(self.texts, workers=2, chunk_size=7)
        phrase = self.searcher.phrase_model.get_phrase('Ontfangen een Missive van')
        matched = [m.phrase for ms in result for m in ms if m.phrase.phrase_string == phrase.phrase_string]
        self.assertTrue(matched)
        self.assertTrue(all(p is phrase for p in matched))

    def test_kwargs_are_passed(self):
        serial = [match_tuples(self.searcher.find_matches(t, include_variants=False)) for t in self.texts]
        result = self.searcher.find_matches_batch(self.texts, workers=2, chunk_size=10, include_variants=False)
        self.assertEqual(serial, [match_tuples(ms) for ms in result])

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            self.searcher.find_matches_batch(self.texts, chunk_size=0)
