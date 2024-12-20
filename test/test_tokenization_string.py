from unittest import TestCase
from fuzzy_search.tokenization.string import make_ngrams, score_char_overlap
from fuzzy_search.tokenization.string import score_ngram_overlap
from fuzzy_search.tokenization.string import score_levenshtein_similarity_ratio
from fuzzy_search.tokenization.string import text2skipgrams
from fuzzy_search.tokenization.string import token2skipgrams


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

    def test_make_ngrams_rejects_zero_size(self):
        error = None
        try:
            make_ngrams('test', 0)
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


class TestToken2SkipGrams(TestCase):

    def test_token2skipgrams_pads_text(self):
        skips = [sg for sg in token2skipgrams('test', ngram_size=2, skip_size=0)]
        self.assertEqual('#t', skips[0].string)

    def test_token2skipgrams_returns_string_if_smaller_than_ngram_size_and_no_padding(self):
        skips = [sg for sg in token2skipgrams('t', ngram_size=2, pad_token=False)]
        self.assertEqual(1, len(skips))

    def test_token2skipgrams_returns_string_if_equal_ngram_size_and_padding(self):
        skips = [sg for sg in token2skipgrams('te', ngram_size=2, pad_token=True)]
        self.assertEqual(5, len(skips))

    def test_token2skipgrams_returns_string_if_equal_ngram_size_and_no_padding(self):
        skips = [sg for sg in token2skipgrams('te', ngram_size=2, pad_token=False)]
        self.assertEqual(1, len(skips))


class TestText2SkipGrams(TestCase):

    def test_text2skipgrams_rejects_n_below_1(self):
        error = None
        try:
            _ = [_ for _ in text2skipgrams('test', ngram_size=0)]
        except ValueError as err:
            error = err
        self.assertNotEqual(None, error)

    def test_text2skipgrams_accepts_n_1(self):
        skips = [sg for sg in text2skipgrams('test', ngram_size=1)]
        self.assertEqual(4, len(skips))

    def test_text2skipgrams_padding_keeps_original_offsets(self):
        text = 'test'
        skips = [sg for sg in text2skipgrams(text, ngram_size=3, skip_size=1)]
        for si, skip in enumerate(skips):
            with self.subTest(si):
                # print(skip.string, skip.start_offset, skip.length, skip.start_offset + skip.length <= len(text))
                self.assertTrue(skip.start_offset + skip.length <= len(text))
        # self.assertEqual(True, False)

    """
    """
    def test_text2skipgrams_returns_string_if_smaller_than_ngram_size(self):
        skips = [sg for sg in text2skipgrams('t', ngram_size=2)]
        self.assertEqual(1, len(skips))

    def test_text2skipgrams_returns_string_if_equal_ngram_size(self):
        skips = [sg for sg in text2skipgrams('te', ngram_size=2)]
        # for skip in skips:
        #     print(skip)
        self.assertEqual(1, len(skips))
