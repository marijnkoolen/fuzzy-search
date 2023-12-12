from unittest import TestCase

import fuzzy_search.similarity as similarity


class TestKeywordList(TestCase):

    def setUp(self) -> None:
        self.keywords = ['test', 'of', 'words' 'of', 'different', 'length']

    def test_keyword_list_accepts_list_of_string(self):
        error = None
        try:
            similarity.KeywordList(self.keywords, max_length_diff=2)
        except BaseException as err:
            error = err
        self.assertEqual(None, error)

    def test_keyword_list_does_not_accept_list_of_dicts(self):
        self.assertRaises(ValueError, similarity.KeywordList, [{'a': 1, 'b': 2}], 2)

    def test_keyword_list_indexes_multiple_occurrences_only_once(self):
        kwl = similarity.KeywordList(self.keywords, max_length_diff=2)
        self.assertEqual(1, kwl.len_keys[2].count('of'))

    def test_keyword_list_indexes_by_length(self):
        kwl = similarity.KeywordList(self.keywords, max_length_diff=2)
        for ti, test_kw in enumerate(self.keywords):
            with self.subTest(ti):
                self.assertIn(test_kw, kwl.len_keys[len(test_kw)])

    def test_keyword_pairs_obey_length_restrictions(self):
        max_length_diff = 2
        kwl = similarity.KeywordList(self.keywords, max_length_diff=max_length_diff)
        for pi, pair in enumerate(kwl.iterate_candidate_pairs()):
            with self.subTest(pi):
                kw1, kw2 = pair
                self.assertEqual(max_length_diff, max(max_length_diff, abs(len(kw1) - len(kw2))))
