from unittest import TestCase
from fuzzy_search.fuzzy_phrase import Phrase
from fuzzy_search.fuzzy_phrase_model import PhraseModel
from fuzzy_search.fuzzy_string import SkipGram
from fuzzy_search.fuzzy_phrase_searcher import FuzzyPhraseSearcher, SkipMatches, Candidate
from fuzzy_search.fuzzy_phrase_searcher import filter_skipgram_threshold, get_skipmatch_candidates

class TestSkipMatches(TestCase):

    def test_skip_matches_registers_match(self):
        skip_matches = SkipMatches(2, 2)
        phrase = Phrase('test')
        skipgram = SkipGram('ts', 0, 3)
        skip_matches.add_skip_match(skipgram, phrase)
        self.assertTrue(phrase in skip_matches.match_set)


class TestCandidate(TestCase):

    def test_candidate_detects_no_match_with_no_skip_match(self):
        phrase = Phrase('test')
        candidate = Candidate(phrase)
        self.assertEqual(candidate.is_match(0.5), False)

    def test_candidate_detects_no_match(self):
        phrase = Phrase('test')
        candidate = Candidate(phrase)
        skipgram = SkipGram('ts', 0, 3)
        candidate.add_skip_match(skipgram)
        self.assertEqual(candidate.is_match(0.5), False)

    def test_candidate_has_skipgram_overlap(self):
        phrase = Phrase('test')
        candidate = Candidate(phrase)
        skipgram = SkipGram('ts', 0, 3)
        candidate.add_skip_match(skipgram)
        self.assertTrue(candidate.get_skip_set_overlap() > 0.0)


class TestFuzzyPhraseSearcher(TestCase):

    def test_can_make_default_phrase_searcher(self):
        searcher = FuzzyPhraseSearcher()
        self.assertNotEqual(searcher, None)

    def test_can_add_phrases_as_strings(self):
        searcher = FuzzyPhraseSearcher()
        phrase = "test"
        searcher.index_phrases(phrases=[phrase])
        phrase_object = searcher.phrases.pop()
        self.assertEqual(phrase_object.phrase_string, phrase)

    def test_can_add_phrases_as_phrase_objects(self):
        searcher = FuzzyPhraseSearcher()
        phrase = Phrase("test")
        searcher.index_phrases(phrases=[phrase])
        self.assertTrue(phrase in searcher.phrases)

    def test_can_generate_skip_matches(self):
        searcher = FuzzyPhraseSearcher()
        phrase = "test"
        searcher.index_phrases(phrases=[phrase])
        text = "this is a test"
        skip_matches = searcher.find_skipgram_matches({"text": text})
        phrase_object = searcher.phrases.pop()
        self.assertTrue(phrase_object in skip_matches.match_set)

    def test_can_filter_skipgram_threshold(self):
        searcher = FuzzyPhraseSearcher()
        phrase = Phrase("test")
        searcher.index_phrases(phrases=[phrase])
        text = "this is a test"
        skip_matches = searcher.find_skipgram_matches({"text": text})
        phrases = filter_skipgram_threshold(skip_matches, 0.5)
        self.assertEqual(len(phrases), 1)

    def test_can_get_candidates(self):
        searcher = FuzzyPhraseSearcher()
        phrase_model = PhraseModel(phrases=["test"])
        searcher.index_phrase_model(phrase_model=phrase_model)
        text = {"text": "this is a test"}
        skip_matches = searcher.find_skipgram_matches(text)
        phrases = get_skipmatch_candidates(text, skip_matches, 0.5, phrase_model=phrase_model)
        self.assertEqual(len(phrases), 1)

    def test_finds_multiple_candidates(self):
        searcher = FuzzyPhraseSearcher()
        phrase_model = PhraseModel(phrases=["test"])
        searcher.index_phrase_model(phrase_model=phrase_model)
        text = {"text": "a test is a test is a test"}
        skip_matches = searcher.find_skipgram_matches(text)
        phrases = get_skipmatch_candidates(text, skip_matches, 0.5, phrase_model=phrase_model)
        self.assertEqual(len(phrases), 3)

    def test_searcher_finds_near_match(self):
        searcher = FuzzyPhraseSearcher()
        phrase = "contains"
        searcher.index_phrases(phrases=[phrase])
        text = "This text consaint some typos."
        matches = searcher.find_matches(text)
        self.assertEqual(len(matches), 1)

    def test_searcher_is_case_sensitive(self):
        searcher = FuzzyPhraseSearcher()
        phrase = "contains"
        searcher.index_phrases(phrases=[phrase])
        text = "This text CONSAINT some typos."
        matches = searcher.find_matches(text)
        self.assertEqual(len(matches), 0)

    def test_searcher_handles_ignorecase(self):
        searcher = FuzzyPhraseSearcher({"ignorecase": True})
        phrase = "contains"
        searcher.index_phrases(phrases=[phrase])
        text = "This text CONSAINT some typos."
        matches = searcher.find_matches(text)
        self.assertEqual(len(matches), 1)

    def test_searcher_uses_word_boundaries(self):
        searcher = FuzzyPhraseSearcher()
        phrase = "contains"
        searcher.index_phrases(phrases=[phrase])
        text = "This text containsi some typos."
        matches = searcher.find_matches(text)
        self.assertEqual(isinstance(matches, list), True)
        self.assertEqual("containsi", matches[0].string)

    def test_searcher_finds_repeat_phrases_as_multiple_matches(self):
        searcher = FuzzyPhraseSearcher()
        phrase = "contains"
        searcher.index_phrases(phrases=[phrase])
        text = "This text contains contains some repetition."
        matches = searcher.find_matches(text)
        self.assertEqual(len(matches), 2)
        self.assertEqual("contains", matches[0].string)
        self.assertEqual("contains", matches[1].string)

    def test_searcher_finds_correct_start(self):
        searcher = FuzzyPhraseSearcher()
        phrase = "contains"
        searcher.index_phrases(phrases=[phrase])
        text = "This text con contains some weirdness."
        matches = searcher.find_matches(text)
        self.assertEqual("contains", matches[0].string)

    def test_searcher_allows_length_variance(self):
        searcher = FuzzyPhraseSearcher()
        phrase = "coffee"
        searcher.index_phrases(phrases=[phrase])
        text = "For sale two units of coffy."
        matches = searcher.find_matches(text)
        self.assertEqual(len(matches), 1)

    def test_searcher_allows_length_variance_2(self):
        searcher = FuzzyPhraseSearcher()
        phrase = "Makelaars"
        searcher.index_phrases(phrases=[phrase])
        text = 'door de Alakei&ers by na gecompletecrt'
        matches = searcher.find_matches(text)
        self.assertEqual(len(matches), 1)

    def test_searcher_can_toggle_variants(self):
        searcher = FuzzyPhraseSearcher({"include_variants": True})
        self.assertEqual(searcher.include_variants, True)

    def test_searcher_can_register_variants(self):
        searcher = FuzzyPhraseSearcher({"include_variants": True})
        phrase = {"phrase": "okay", "variants": ["OK"]}
        searcher.index_phrase_model(phrase_model=PhraseModel([phrase]))
        self.assertEqual(len(searcher.variants), 1)
        variant = searcher.variants.pop()
        self.assertEqual(variant.phrase_string, "OK")

    def test_searcher_can_match_variants(self):
        searcher = FuzzyPhraseSearcher({"include_variants": True})
        phrase = {"phrase": "okay", "variants": ["OK"]}
        searcher.index_phrase_model(phrase_model=PhraseModel([phrase]))
        text = "This text is okay and this test is OK."
        matches = searcher.find_matches(text, include_variants=True)
        self.assertEqual(matches[1].phrase.phrase_string, phrase["phrase"])
        self.assertEqual(matches[1].variant.phrase_string, phrase["variants"][0])

    def test_searcher_can_toggle_distractors(self):
        searcher = FuzzyPhraseSearcher({"filter_distractors": True})
        self.assertEqual(searcher.filter_distractors, True)

    def test_searcher_can_register_distractors(self):
        searcher = FuzzyPhraseSearcher({"filter_distractors": True})
        phrase = {"phrase": "okay", "distractors": ["OK"]}
        searcher.index_phrase_model(phrase_model=PhraseModel([phrase]))
        self.assertEqual(len(searcher.distractors), 1)
        distractor = searcher.distractors.pop()
        self.assertEqual(distractor.phrase_string, "OK")

    def test_searcher_can_match_distractors(self):
        searcher = FuzzyPhraseSearcher({"filter_distractors": True})
        phrase = {"phrase": "baking", "distractors": ["braking"]}
        searcher.index_phrase_model(phrase_model=PhraseModel([phrase]))
        text = "This text is about baking and not about braking."
        matches = searcher.find_matches(text, filter_distractors=True)
        self.assertEqual(len(matches), 1)
    """


"""
class TestFuzzySearchExactMatch(TestCase):

    def setUp(self) -> None:
        self.searcher = FuzzyPhraseSearcher()
        self.phrase = {"phrase": "baking", "distractors": ["braking"]}
        self.searcher.index_phrase_model(phrase_model=PhraseModel([self.phrase]))

    def test_fuzzy_search_can_search_exact_match(self):
        text = "This text is about baking and not about braking."
        matches = self.searcher.find_exact_matches(text)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].string, "baking")

    def test_fuzzy_search_can_search_exact_match_with_word_boundaries(self):
        text = "This text is about baking and not about rebaking."
        matches = self.searcher.find_exact_matches(text, use_word_boundaries=True)
        self.assertEqual(len(matches), 1)

    def test_fuzzy_search_can_search_exact_match_without_word_boundaries(self):
        text = "This text is about baking and not about rebaking."
        matches = self.searcher.find_exact_matches(text, use_word_boundaries=False)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[1].string, "baking")

    def test_fuzzy_search_can_search_exact_match_with_special_characters(self):
        searcher = FuzzyPhraseSearcher()
        phrase = {"phrase": "[baking]", "distractors": ["braking"]}
        searcher.index_phrase_model(phrase_model=PhraseModel([phrase]))
        text = "This text is about [baking] and not about braking."
        matches = searcher.find_exact_matches(text, use_word_boundaries=False)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].string, "[baking]")

    def test_text_split(self):
        text = {
            "text": "Ntfangen een Missive van den Gouverneur Generaal van het eiland Amoras, "
                    + "verfoekende, dat dit beter getest moet worden.",
            "id": "urn:republic/inv=3825/meeting=1775-01-28/para=2"
        }
        phrases = [
            {"phrase": "ONtfangen een Missive van"},
            {"phrase": "Missive"},
            {"phrase": "Gouverneur Generaal"},
            {"phrase": "Gouverneur"},
            {"phrase": "Generaal van de"},
            {"phrase": "versoekende"}
        ]
        phrase_model = PhraseModel(model=phrases)
        searcher = FuzzyPhraseSearcher()
        searcher.index_phrase_model(phrase_model)
        exact_matches = searcher.find_exact_matches(text)
        self.assertEqual(len(exact_matches), 3)


class TestSearcherRealData(TestCase):

    def setUp(self) -> None:
        self.text1 = "ie Veucris den 5. Januaris 1725. PR&ASIDE, Den Heere Bentinck. PRASENTIEBUS, " + \
                     "De Heeren Jan Welderen , van Dam, Torck , met een extraordinaris Gedeputeerde" + \
                     " uyt de Provincie van Gelderlandt. Van Maasdam , vanden Boeizelaar , Raadtpen" + \
                     "fionaris van Hoornbeeck , met een extraordinaris Gedeputeerde uyt de Provinci" + \
                     "e van Hollandt ende Welt-Vrieslandt. Velters, Ockere , Noey; van Hoorn , met " + \
                     "een extraordinaris Gedeputeerde uyt de Provincie van Zeelandt. Van Renswoude " + \
                     ", van Voor{t. Van Schwartzenbergh, vander Waayen, Vegilin Van I{elmuden. Van " + \
                     "Iddekinge ‚ van Tamminga."
        self.text2 = "Mercuri: den 10. Jangarii, 1725. ia PRESIDE, Den Heere an Iddekinge. PRA&SENT" + \
                     "IBUS, De Heeren /an Welderen , van Dam, van Wynbergen, Torck, met een extraor" + \
                     "dinaris Gedeputeerde uyt de Provincie van Gelderland. Van Maasdam , Raadtpenf" + \
                     "ionaris van Hoorn=beeck. Velters, Ockerfe, Noey. Taats van Amerongen, van Ren" + \
                     "swoude. Vander Waasen , Vegilin, ’ Bentinck, van I(elmaden. Van Tamminga."
        self.config = {
            "char_match_threshold": 0.6,
            "ngram_threshold": 0.5,
            "levenshtein_threshold": 0.6,
            "ignorecase": False,
            "max_length_variance": 3,
            "ngram_size": 2,
            "skip_size": 2,
        }
        self.searcher = FuzzyPhraseSearcher(self.config)
        # create a list of domain phrases
        self.domain_phrases = [
            # terms for the chair and attendants of a meeting
            "PRAESIDE",
            "PRAESENTIBUS",
            # some weekdays in Latin
            "Veneris",
            "Mercurii",
            # some date phrase where any date in January 1725 should match
            "den .. Januarii 1725"
        ]
        self.phrase_model = PhraseModel(phrases=self.domain_phrases)
        # register the keywords with the searcher
        self.searcher.index_phrase_model(self.phrase_model)

    def test_fuzzy_search_text1_finds_four_matches(self):
        matches = self.searcher.find_matches(self.text1)
        self.assertEqual(len(matches), 4)

    def test_fuzzy_search_text1_finds_friday(self):
        matches = self.searcher.find_matches(self.text1)
        self.assertEqual(matches[0].string, "Veucris")

    def test_fuzzy_search_text1_finds_date(self):
        matches = self.searcher.find_matches(self.text1)
        self.assertEqual(matches[1].string, "den 5. Januaris 1725")

    def test_fuzzy_search_text1_finds_president(self):
        matches = self.searcher.find_matches(self.text1)
        self.assertEqual(matches[2].string, "PR&ASIDE")

    def test_fuzzy_search_text1_finds_attendants(self):
        matches = self.searcher.find_matches(self.text1)
        self.assertEqual(matches[3].string, "PRASENTIEBUS")

    def test_fuzzy_search_text2_finds_four_matches(self):
        matches = self.searcher.find_matches(self.text2)
        self.assertEqual(len(matches), 4)

    def test_fuzzy_search_text2_finds_friday(self):
        matches = self.searcher.find_matches(self.text2)
        self.assertEqual(matches[0].string, "Mercuri")

    def test_fuzzy_search_text2_finds_date(self):
        matches = self.searcher.find_matches(self.text2)
        self.assertEqual(matches[1].string, "den 10. Jangarii, 1725")

    def test_fuzzy_search_text2_finds_president(self):
        matches = self.searcher.find_matches(self.text2)
        self.assertEqual(matches[2].string, "PRESIDE")

    def test_fuzzy_search_text2_finds_attendants(self):
        matches = self.searcher.find_matches(self.text2)
        self.assertEqual(matches[3].string, "PRA&SENTIBUS")


class TestSearcherRealData2(TestCase):

    def setUp(self) -> None:
        self.text1 = 'TS gehoort het rapport van de Heeren I van Lynden'
        self.config = {
            "char_match_threshold": 0.7,
            "ngram_threshold": 0.7,
            "levenshtein_threshold": 0.7,
            "ignorecase": False,
            "include_variants": True,
            "max_length_variance": 3,
            "ngram_size": 2,
            "skip_size": 2,
        }
        self.searcher = FuzzyPhraseSearcher(self.config)
        # create a list of domain phrases
        self.domain_phrases = [
            {
                'phrase': 'den Heere',
                'variants': [
                    'de Heer',
                    'de Heeren',
                ]
            }
        ]
        self.phrase_model = PhraseModel(phrases=self.domain_phrases)
        # register the keywords with the searcher
        self.searcher.index_phrase_model(self.phrase_model)

    def test_searcher_find_no_overlapping_variants(self):
        phrase_matches = self.searcher.find_matches(self.text1)
        self.assertEqual(len(phrase_matches), 1)
