from unittest import TestCase
import fuzzy_search.tokenization.subtoken as subtoken


class TestCorpus(TestCase):

    def setUp(self) -> None:
        token_sets = [
            ['low'] * 5,
            ['lowest'] * 2,
            ['newer'] * 6,
            ['wider'] * 3,
            ['new'] * 2
        ]
        self.tokens = [token for token_set in token_sets for token in token_set]

    def test_corpus_contains_bpetokens(self):
        corpus = subtoken.string_tokens_to_corpus(self.tokens)
        for ti, token in enumerate(corpus):
            with self.subTest(ti):
                self.assertEqual(True, isinstance(token, subtoken.BPEToken))

    def test_find_new_symbol_pairs(self):
        merge_symbol = 'er'
        subtests = [
            {
                'token': ('n', 'e', 'w', 'er'),
                'expected_pairs': [('w', 'er')]
            },
            {
                'token': ('n', 'e', 'w', 'er', ' '),
                'expected_pairs': [('w', 'er'), ('er', ' ')]
            },
            {
                'token': ('er', ' '),
                'expected_pairs': [('er', ' ')]
            }
        ]
        for si, subtest in enumerate(subtests):
            with self.subTest(si):
                new_pairs = subtoken.find_new_symbol_pairs(merge_symbol, subtest['token'])
                self.assertEqual(subtest['expected_pairs'], new_pairs)

    def test_generate_token_symbol_pair(self):
        token = tuple(['n', 'e', 'w', 'e', 'r'])
        expected_list = [('n', 'e'), ('e', 'w'), ('w', 'e'), ('e', 'r')]
        symbol_pair_list = subtoken.generate_symbol_pairs(token)
        self.assertEqual(expected_list, symbol_pair_list)

    def test_compare_symbol_pairs(self):
        token1 = tuple(['n', 'e', 'w', 'e', 'r'])
        token2 = tuple(['n', 'e', 'w', 'er'])
        overlap, only1, only2 = subtoken.compare_token_symbol_pairs(token1, token2)
        self.assertEqual({('w', 'e'), ('e', 'r')}, only1)
        self.assertEqual({('w', 'er')}, only2)

    def test_merge_order(self):
        merge_order = [
            ('er', 'r '), ('er', 'er '),
            ('ne', 'ew'), ('ne', 'ew'),
            'new', ('lo', 'ow'), 'low', 'newer ', 'low '
        ]
        for mi, merge in enumerate(merge_order):
            with self.subTest(mi):
                k = mi+1
                print(f"\nk: {k}")
                vocab = subtoken.make_byte_pair_encoding(self.tokens, k)
                if isinstance(merge, str):
                    self.assertEqual(True, merge in vocab)
                elif isinstance(merge, tuple):
                    self.assertEqual(True, any(m in vocab for m in merge))


class TestFrequencyTracker(TestCase):

    def setUp(self):
        self.tracker = subtoken.FrequencyTracker()

    def test_basic_update_and_frequency(self):
        self.tracker.update(("ap", "ple"), 2)
        self.tracker.update(("ban", "ana"), 1)
        self.assertEqual(self.tracker.frequency_of(("ap", "ple")), 2)
        self.assertEqual(self.tracker.frequency_of(("ban", "ana")), 1)
        self.assertEqual(self.tracker.frequency_of("pie"), 0)

    def test_negative_update_removes_element(self):
        self.tracker.update(("ap", "ple"), 3)
        self.tracker.update(("ap", "ple"), -3)
        self.assertEqual(self.tracker.frequency_of(("ap", "ple")), 0)
        self.assertIsNone(self.tracker.most_frequent())

    def test_most_frequent(self):
        self.tracker.update(("ap", "ple"), 1)
        self.tracker.update(("ban", "ana"), 3)
        self.tracker.update(("p", "ie"), 2)
        self.assertEqual(self.tracker.most_frequent(), (('ban', 'ana'), 3))

    def test_all_with_max_frequency(self):
        self.tracker.update(("ap", "ple"), 2)
        self.tracker.update(("p", "ie"), 2)
        self.tracker.update(("ja", "m"), 1)
        self.assertSetEqual(self.tracker.all_with_max_frequency(), {('ap', 'ple'), ('p', 'ie')})
        self.assertSetEqual(self.tracker.all_with_max_frequency(length=3), {('p', 'ie')})

    def test_most_frequent_with_length_filter(self):
        self.tracker.update(("ap", "ple"), 2)
        self.tracker.update(("p", "ear"), 2)
        self.tracker.update(("fi", "g"), 1)
        self.assertEqual(self.tracker.most_frequent(length=4), (('p', 'ear'), 2))
        self.assertEqual(self.tracker.most_frequent(length=5), (('ap', 'ple'), 2))
        self.assertIsNone(self.tracker.most_frequent(length=3))  # no 3-letter word with freq 2

    def test_most_frequent_shortest(self):
        self.tracker.update(("ap", "ple"), 3)   # len = 5
        self.tracker.update(("fi", "g"), 3)     # len = 3
        self.tracker.update(("p", "ear"), 2)    # len = 4
        result = self.tracker.most_frequent_shortest()
        self.assertIn(result[0], {('fi', 'g')})
        self.assertEqual(result[1], 3)
        self.assertEqual(result[2], 3)

    def test_zero_and_negative_updates(self):
        self.tracker.update(("ap", "ple"), 0)
        self.tracker.update(("ban", "ana"), -1)
        self.assertEqual(self.tracker.frequency_of(("ap", "ple")), 0)
        self.assertEqual(self.tracker.frequency_of(("ban", "ana")), 0)

    def test_update_to_below_zero_removes(self):
        self.tracker.update(("p", "ear"), 1)
        self.tracker.update(("p", "ear"), -5)
        self.assertEqual(self.tracker.frequency_of(("p", "ear")), 0)
        self.assertIsNone(self.tracker.most_frequent())

    def test_tie_breaking_length(self):
        self.tracker.update(("straw", "berry"), 5)  # length 10
        self.tracker.update(("ja", "m"), 5)         # length 3
        self.tracker.update(("to", "ast"), 5)       # length 5
        result = self.tracker.most_frequent_shortest()
        self.assertEqual(result[1], 5)
        self.assertEqual(result[2], 3)
        self.assertIn(result[0], {("ja", "m")})
