from unittest import TestCase

from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.phrase.phrase_model import PhraseModel
from fuzzy_search.search.phrase_searcher import FuzzyPhraseSearcher
from fuzzy_search.match.skip_match import filter_skipgram_threshold
from fuzzy_search.match.skip_match import get_skipmatch_candidates
from fuzzy_search.tokenization.token import Tokenizer


class TestFuzzyPhraseSearcher(TestCase):

    def setUp(self) -> None:
        self.searcher = FuzzyPhraseSearcher()

    def test_can_make_default_phrase_searcher(self):
        self.assertNotEqual(self.searcher, None)

    def test_can_add_phrases_as_strings(self):
        phrase = "test"
        self.searcher.index_phrases(phrases=[phrase])
        phrase_object = self.searcher.phrases.pop()
        self.assertEqual(phrase_object.phrase_string, phrase)

    def test_can_add_phrases_as_phrase_objects(self):
        phrase = Phrase("test")
        self.searcher.index_phrases(phrases=[phrase])
        self.assertTrue(phrase in self.searcher.phrases)

    def test_can_generate_skip_matches(self):
        phrase = "test"
        self.searcher.index_phrases(phrases=[phrase])
        text = "this is a test"
        skip_matches = self.searcher.find_skipgram_matches({"text": text})
        phrase_object = self.searcher.phrases.pop()
        self.assertTrue(phrase_object in skip_matches.match_set)

    def test_can_filter_skipgram_threshold(self):
        phrase = Phrase("test")
        self.searcher.index_phrases(phrases=[phrase])
        text = "this is a test"
        skip_matches = self.searcher.find_skipgram_matches({"text": text})
        phrases = filter_skipgram_threshold(skip_matches, 0.5)
        self.assertEqual(len(phrases), 1)

    def test_can_get_candidates(self):
        phrase_model = PhraseModel(phrases=["test"])
        self.searcher.index_phrase_model(phrase_model=phrase_model)
        text = {"text": "this is a test"}
        skip_matches = self.searcher.find_skipgram_matches(text)
        phrases = get_skipmatch_candidates(text, skip_matches, 0.5, phrase_model=phrase_model)
        self.assertEqual(len(phrases), 1)

    def test_finds_multiple_candidates(self):
        phrase_model = PhraseModel(phrases=["test"])
        self.searcher.index_phrase_model(phrase_model=phrase_model)
        text = {"text": "a test is a test is a test"}
        skip_matches = self.searcher.find_skipgram_matches(text)
        phrases = get_skipmatch_candidates(text, skip_matches, 0.5, phrase_model=phrase_model)
        self.assertEqual(len(phrases), 3)

    def test_searcher_finds_near_match(self):
        phrase = "contains"
        self.searcher.index_phrases(phrases=[phrase])
        text = "This text consaint some typos."
        matches = self.searcher.find_matches(text)
        self.assertEqual(len(matches), 1)

    def test_searcher_is_case_sensitive(self):
        phrase = "contains"
        self.searcher.index_phrases(phrases=[phrase])
        text = "This text CONSAINT some typos."
        matches = self.searcher.find_matches(text)
        self.assertEqual(len(matches), 0)

    def test_searcher_handles_ignorecase(self):
        searcher = FuzzyPhraseSearcher(config={"ignorecase": True})
        phrase = "contains"
        searcher.index_phrases(phrases=[phrase])
        text = "This text CONSAINT some typos."
        matches = searcher.find_matches(text)
        self.assertEqual(len(matches), 1)

    def test_searcher_uses_word_boundaries(self):
        phrase = "contains"
        self.searcher.index_phrases(phrases=[phrase])
        text = "This text containsi some typos."
        matches = self.searcher.find_matches(text)
        self.assertEqual(isinstance(matches, list), True)
        self.assertEqual("containsi", matches[0].string)

    def test_searcher_finds_repeat_phrases_as_multiple_matches(self):
        phrase = "contains"
        self.searcher.index_phrases(phrases=[phrase])
        text = "This text contains contains some repetition."
        matches = self.searcher.find_matches(text)
        self.assertEqual(len(matches), 2)
        self.assertEqual("contains", matches[0].string)
        self.assertEqual("contains", matches[1].string)

    def test_searcher_finds_correct_start(self):
        phrase = "contains"
        self.searcher.index_phrases(phrases=[phrase])
        text = "This text con contains some weirdness."
        matches = self.searcher.find_matches(text)
        self.assertEqual("contains", matches[0].string)

    def test_searcher_allows_length_variance(self):
        phrase = "coffee"
        self.searcher.index_phrases(phrases=[phrase])
        text = "For sale two units of coffy."
        matches = self.searcher.find_matches(text)
        self.assertEqual(len(matches), 1)

    def test_searcher_allows_length_variance_2(self):
        phrase = "Makelaars"
        self.searcher.index_phrases(phrases=[phrase])
        text = 'door de Alakei&ers by na gecompletecrt'
        matches = self.searcher.find_matches(text, debug=4)
        self.assertEqual(1, len(matches))

    def test_searcher_accepts_phrases_on_init(self):
        phrase = "Makelaars"
        searcher = FuzzyPhraseSearcher(phrase_list=[phrase])
        self.assertEqual(1, len(searcher.phrases))

    def test_searcher_accepts_tokenized_document(self):
        phrase = "Makelaars"
        self.searcher.index_phrases(phrases=[phrase])
        text = 'door de Alakei&ers by na gecompletecrt'
        tokenizer = Tokenizer()
        doc = tokenizer.tokenize_doc(text)
        error = None
        try:
            matches = self.searcher.find_matches(doc)
        except TypeError as err:
            error = err
        self.assertEqual(None, error)
        self.assertEqual(1, len(matches))


class TestFuzzySearchVariants(TestCase):

    def setUp(self) -> None:
        self.searcher = FuzzyPhraseSearcher(config={"include_variants": True})
        self.phrase = {"phrase": "okay", "variants": ["OK"]}

    def test_searcher_can_toggle_variants(self):
        self.assertEqual(self.searcher.include_variants, True)

    def test_searcher_can_register_variants(self):
        self.searcher.index_phrase_model(phrase_model=PhraseModel([self.phrase]))
        self.assertEqual(len(self.searcher.variants), 1)
        variant = self.searcher.variants.pop()
        self.assertEqual(variant.phrase_string, "OK")

    def test_searcher_can_match_variants(self):
        self.searcher.index_phrase_model(phrase_model=PhraseModel([self.phrase]))
        text = "This text is okay and this test is OK."
        matches = self.searcher.find_matches(text, include_variants=True)
        self.assertEqual(matches[1].phrase.phrase_string, self.phrase["phrase"])
        self.assertEqual(matches[1].variant.phrase_string, self.phrase["variants"][0])


class TestFuzzySearchDistractors(TestCase):

    def setUp(self) -> None:
        self.searcher = FuzzyPhraseSearcher(config={"filter_distractors": True})
        self.phrase = {"phrase": "okay", "distractors": ["OK"]}

    def test_searcher_can_toggle_distractors(self):
        self.assertEqual(self.searcher.filter_distractors, True)

    def test_searcher_can_register_distractors(self):
        self.searcher.index_phrase_model(phrase_model=PhraseModel([self.phrase]))
        self.assertEqual(len(self.searcher.distractors), 1)
        distractor = self.searcher.distractors.pop()
        self.assertEqual(distractor.phrase_string, "OK")

    def test_searcher_can_match_distractors(self):
        phrase = {"phrase": "baking", "distractors": ["braking"]}
        self.searcher.index_phrase_model(phrase_model=PhraseModel([phrase]))
        text = "This text is about baking and not about braking."
        matches = self.searcher.find_matches(text, filter_distractors=True)
        self.assertEqual(len(matches), 1)


class TestFuzzySearchExactMatch(TestCase):

    def setUp(self) -> None:
        self.searcher = FuzzyPhraseSearcher()
        self.phrase = {"phrase": "baking", "distractors": ["braking"]}
        self.searcher.index_phrase_model(phrase_model=PhraseModel([self.phrase]))

    def test_fuzzy_search_can_search_exact_match(self):
        text = "This text is about baking and not about braking."
        matches = self.searcher.find_exact_matches(text, debug=True)
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
        self.searcher = FuzzyPhraseSearcher(config=self.config)
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

    def test_fuzzy_search_text1_finds_five_matches(self):
        matches = self.searcher.find_matches(self.text1)
        # for mi, match in enumerate(matches):
        #     print(f"finds_four_matches - match {mi}:", match)
        self.assertEqual(5, len(matches))

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

    def test_fuzzy_search_text2_finds_five_matches(self):
        matches = self.searcher.find_matches(self.text2)
        # for mi, match in enumerate(matches):
        #     print(f"finds_four_matches - match {mi}:", match)
        self.assertEqual(5, len(matches))

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
            "char_match_threshold": 0.6,
            "ngram_threshold": 0.5,
            "levenshtein_threshold": 0.6,
            "ignorecase": False,
            "include_variants": True,
            "max_length_variance": 3,
            "ngram_size": 2,
            "skip_size": 2,
        }
        self.searcher = FuzzyPhraseSearcher(config=self.config)
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

    def test_searcher_allows_length_variance_2(self):
        searcher = FuzzyPhraseSearcher(config=self.config)
        searcher.ignorecase = True
        phrase = "Admiraliteiten in t gemeen"
        searcher.index_phrases(phrases=[phrase])
        text = 'aaniraliteyten in het gemeen'
        matches = searcher.find_matches(text)
        self.assertEqual(len(matches), 1)

    def test_searcher_allows_length_variance_3(self):
        searcher = FuzzyPhraseSearcher(config=self.config)
        searcher.ignorecase = True
        phrase = 'Admiraliteit in Vriesland'
        searcher.index_phrases(phrases=[phrase])
        text = 'AduiraliteytVrieslaidt'
        matches = searcher.find_matches(text)
        self.assertEqual(len(matches), 1)

    def test_searcher_finds_DONtfangen(self):
        searcher = FuzzyPhraseSearcher(config=self.config)
        searcher.ignorecase = True
        phrase = "ONtfangen een Missive van"
        searcher.index_phrases(phrases=[phrase])
        text = 'DONtfangen een Missive van den Heere vander Goes'
        matches = searcher.find_matches(text)
        self.assertEqual(len(matches), 1)

    def test_searcher_finds_long_opening(self):
        searcher = FuzzyPhraseSearcher(config=self.config)
        searcher.ignorecase = True
        phrases = [
            "hebben ter Vergaderinge ingebraght",
            'hebben ter Vergaderinge ingebragt en laaten leezen de Resolutie'
        ]
        searcher.index_phrases(phrases=phrases)
        text = "De Heeren Gedeputeerden van de Provincie van Zeelandt, hebben ter Vergaderinge ingebraght en laten lesen de Resolutie van de Heeren Staten van de hoogh-gemelde Provincie hare Principalen, raeckende het negotieren van hare quote voor een derde part in de Petitie tot de extraordinaris Equipage voor het loopende jaer, volgende de voorschreve Resolutie hier na geinsereert."
        matches = searcher.find_matches(text)
        self.assertEqual(len(matches), 2)

