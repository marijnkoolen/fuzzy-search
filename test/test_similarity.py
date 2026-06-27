import math
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


class TestSkipgramSimilarity(TestCase):

    def setUp(self) -> None:
        self.terms = ['huygens', 'huygens', 'huygensia', 'kortjakje', 'wordeling', 'amsterdam', 'rotterdam']
        self.sim = similarity.SkipgramSimilarity(ngram_length=2, skip_length=1, terms=self.terms,
                                                  max_length_diff=2)

    def test_indexes_unique_terms(self):
        self.assertEqual(len(set(self.terms)), len(self.sim.vocabulary))

    def test_identical_term_has_similarity_one(self):
        top_terms = self.sim.rank_similar('huygens', top_n=1, score_cutoff=0.0)
        self.assertEqual('huygens', top_terms[0][0])
        self.assertAlmostEqual(1.0, top_terms[0][1], places=6)

    def test_rank_similar_returns_sorted_descending_scores(self):
        top_terms = self.sim.rank_similar('huygens', top_n=10, score_cutoff=0.0)
        scores = [score for _, score in top_terms]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rank_similar_respects_top_n(self):
        top_terms = self.sim.rank_similar('huygens', top_n=2, score_cutoff=0.0)
        self.assertLessEqual(len(top_terms), 2)

    def test_rank_similar_respects_score_cutoff(self):
        top_terms = self.sim.rank_similar('huygens', top_n=10, score_cutoff=1.1)
        self.assertEqual([], top_terms)

    def test_rank_similar_respects_max_length_diff(self):
        # 'huygens' (7 chars) vs 'huygensia' (9 chars) differ by 2, so a
        # max_length_diff of 1 should exclude 'huygensia' from the results.
        sim = similarity.SkipgramSimilarity(ngram_length=2, skip_length=1, terms=self.terms,
                                            max_length_diff=1)
        top_terms = sim.rank_similar('huygens', top_n=10, score_cutoff=0.0)
        matched_terms = {term for term, _ in top_terms}
        self.assertNotIn('huygensia', matched_terms)

    def test_unknown_term_returns_empty_list_for_high_cutoff(self):
        top_terms = self.sim.rank_similar('zzzzzzz', top_n=10, score_cutoff=0.5)
        self.assertEqual([], top_terms)

    def test_dot_product_only_visits_in_range_length_buckets(self):
        # The whole point of bucketing by term length is to avoid ever
        # multiplying the query against terms outside max_length_diff.
        # Buckets are built lazily on demand, so after a single query only
        # in-range lengths should ever have been materialized.
        sim = similarity.SkipgramSimilarity(
            ngram_length=2, skip_length=1,
            terms=self.terms + ['hi', 'a-very-long-and-unrelated-term'],
            max_length_diff=2,
        )
        query_length = len('huygens')
        all_lengths = set(sim._term_ids_by_length)
        in_range = {length for length in all_lengths if abs(length - query_length) <= sim.max_length_diff}
        out_of_range = all_lengths - in_range
        self.assertTrue(out_of_range, "test setup should include terms outside the length range")
        self.assertTrue(in_range, "test setup should include terms inside the length range")

        self.assertEqual({}, sim._length_buckets, "buckets should not be built before any query")
        sim._compute_dot_product('huygens')

        self.assertTrue(set(sim._length_buckets) & in_range)
        self.assertFalse(set(sim._length_buckets) & out_of_range,
                          f"built out-of-range length buckets: {set(sim._length_buckets) & out_of_range}")

    def test_indexing_more_terms_does_not_rebuild_unrelated_buckets(self):
        # Adding new terms of one length should not invalidate or force a
        # rebuild of an already-built bucket for a different length.
        sim = similarity.SkipgramSimilarity(ngram_length=2, skip_length=1,
                                            terms=['huygens', 'amsterdam'], max_length_diff=2)
        sim._compute_dot_product('huygens')  # builds buckets for lengths near 7
        built_matrix_for_9 = sim._length_buckets[len('amsterdam')]

        sim.index_terms(['rotterdam'], reset_index=False)  # also length 9, should mark length 9 dirty only
        self.assertIn(len('rotterdam'), sim._dirty_lengths)
        self.assertNotIn(len('huygens'), sim._dirty_lengths)
        # the length-9 bucket matrix object itself hasn't been rebuilt yet (still the old one)
        self.assertIs(built_matrix_for_9, sim._length_buckets[len('amsterdam')])

        sim._compute_dot_product('amsterdam')  # forces a rebuild of just the length-9 bucket
        self.assertNotIn(len('rotterdam'), sim._dirty_lengths)
        self.assertIsNot(built_matrix_for_9, sim._length_buckets[len('amsterdam')])
        self.assertEqual(2, sim._length_buckets[len('amsterdam')].shape[1])

    def test_index_terms_with_reset_replaces_index(self):
        self.sim.index_terms(['wordeling'], reset_index=True)
        self.assertEqual(1, len(self.sim.vocabulary))
        top_terms = self.sim.rank_similar('wordeling', top_n=1, score_cutoff=0.0)
        self.assertEqual('wordeling', top_terms[0][0])

    def test_index_terms_without_reset_extends_index(self):
        sim = similarity.SkipgramSimilarity(ngram_length=2, skip_length=1, terms=['huygens'],
                                            max_length_diff=2)
        sim.index_terms(['amsterdam'], reset_index=False)
        self.assertEqual(2, len(sim.vocabulary))

    def test_matches_brute_force_cosine_similarity(self):
        # Cross-check the vectorised implementation against a direct,
        # unoptimised recomputation of skipgram cosine similarity.
        query = 'huygens'
        query_freq = self.sim._term_to_skip(query)
        query_vl = math.sqrt(sum(f ** 2 for f in query_freq.values()))
        expected = {}
        for term in set(self.terms):
            if abs(len(term) - len(query)) > self.sim.max_length_diff:
                continue
            term_freq = self.sim._term_to_skip(term)
            shared = set(query_freq) & set(term_freq)
            dot = sum(query_freq[s] * term_freq[s] for s in shared)
            if dot == 0:
                continue
            term_vl = math.sqrt(sum(f ** 2 for f in term_freq.values()))
            expected[term] = dot / (query_vl * term_vl)
        top_terms = dict(self.sim.rank_similar(query, top_n=10, score_cutoff=0.0))
        self.assertEqual(set(expected.keys()), set(top_terms.keys()))
        for term, score in expected.items():
            with self.subTest(term):
                self.assertAlmostEqual(score, top_terms[term], places=6)
