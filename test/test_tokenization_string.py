from unittest import TestCase
from fuzzy_search.tokenization.string import make_ngrams, score_char_overlap
from fuzzy_search.tokenization.string import score_ngram_overlap
from fuzzy_search.tokenization.string import score_levenshtein_similarity_ratio


class Test(TestCase):

    ###############
    # make_ngrams #
    ###############

    def test_make_ngrams_rejects_non_string_text(self):
        error = None
        try:
            make_ngrams(1, 1.5)
        except TypeError as err:
            error = err
        self.assertNotEqual(error, None)

    def test_make_ngrams_rejects_non_integer_size(self):
        error = None
        try:
            make_ngrams('test', 1.5)
        except TypeError as err:
            error = err
        self.assertNotEqual(error, None)

    def test_make_ngrams_rejects_negative_size(self):
        error = None
        try:
            make_ngrams('test', -1)
        except ValueError as err:
            error = err
        self.assertNotEqual(error, None)

    def test_make_ngrams_accepts_positive_integer(self):
        error = None
        try:
            make_ngrams('test', 1)
        except ValueError as err:
            error = err
        self.assertEqual(error, None)

    def test_make_ngrams_handles_size_one_correctly(self):
        text = 'test'
        ngram_size = 1
        ngrams = make_ngrams(text, ngram_size)
        num_ngrams = len(text) + 3 - ngram_size
        self.assertEqual(len(ngrams), num_ngrams)

    def test_make_ngrams_handles_size_two_correctly(self):
        text = 'test'
        ngram_size = 2
        ngrams = make_ngrams(text, ngram_size)
        num_ngrams = len(text) + 3 - ngram_size
        self.assertEqual(len(ngrams), num_ngrams)

    def test_make_ngrams_rejects_integer_larger_than_text_length(self):
        ngrams = make_ngrams('test', 10)
        self.assertEqual(len(ngrams), 0)

    #######################
    # score_ngram_overlap #
    #######################

    def test_score_ngram_overlap_is_num_ngrams_for_self_comparison(self):
        text = 'test'
        ngram_size = 2
        ngrams = make_ngrams(text, ngram_size)
        overlap = score_ngram_overlap(text, text, ngram_size)
        self.assertEqual(overlap, len(ngrams))

    def test_score_ngram_overlap_is_zero_for_comparison_with_empty(self):
        text = 'test'
        ngram_size = 2
        overlap = score_ngram_overlap(text, '', ngram_size)
        self.assertEqual(overlap, 0)

    ######################
    # score_char_overlap #
    ######################

    def test_score_char_overlap_with_self_is_len_of_self(self):
        text = 'test'
        overlap = score_char_overlap(text, text)
        self.assertEqual(overlap, len(text))

    def test_score_char_overlap_with_smaller_word_is_smaller_than_len_of_self(self):
        text = 'ttttt'
        overlap = score_char_overlap(text, text[:4])
        self.assertNotEqual(overlap, len(text))

    ################################
    # score_levenshtein_similarity #
    ################################

    def test_score_levenshtein_similarity_with_self_is_len_of_self(self):
        text = 'test'
        similarity = score_levenshtein_similarity_ratio(text, text)
        self.assertEqual(similarity, 1)
