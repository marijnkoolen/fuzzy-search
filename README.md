# fuzzy-search
Fuzzy search modules for searching occurrences of words or phrases in low quality OCR and HTR text and text with spelling variation. 

This package has been developed for finding occurrences of formulaic phrases in corpora with repetitive texts, such as auction advertisements in 17th century newspapers, resolutions in political administrative archives, or notarial deeds.

The basic fuzzy searcher uses character skip-grams to exhaustively search for approximately matching strings for a given list of target words and phrases. 

## Usage

Below is an example of searching for key words and phrases in the corpus of resolutions of the States General of the Dutch Republic. 


```python
from fuzzy_search import FuzzyPhraseSearcher

# highger matching thresholds for higher quality OCR/HTR (higher precision, recall should be good anyway)
# lower matching thresholds for lower quality OCR/HTR (higher recall, as that's the main problem)

config = {
    "char_match_threshold": 0.6,
    "ngram_threshold": 0.5,
    "levenshtein_threshold": 0.6,
    "ignorecase": False,
    "max_length_variance": 3,
    "ngram_size": 2,
    "skip_size": 2,
}

# create a list of domain- or corpus-specific terms and formulaic phrases
domain_phrases = [
    # terms for announcing the chair and attendants of a meeting
    "PRAESIDE",
    "PRAESENTIBUS",
    # some weekdays in Latin
    "Veneris", 
    "Mercurii",
    # some date phrase where any date in January 1725 should match
    "den .. Januarii 1725"
]

# initialize a new searcher instance with the phrases and the config
fuzzy_searcher = FuzzyPhraseSearcher(config=config, phrase_model=domain_phrases)

# take some example texts: resolutoins of the meeting of the Dutch States General on
# Friday the 5th of January, 1725
text = ("ie Veucris den 5. Januaris 1725. PR&ASIDE, Den Heere Bentinck. PRASENTIEBUS, "
        "De Heeren Jan Welderen , van Dam, Torck , met een extraordinaris Gedeputeerde "
        "uyt de Provincie van Gelderlandt. Van Maasdam , vanden Boeizelaar , Raadtpenfionaris "
        "van Hoornbeeck , met een extraordinaris Gedeputeerde uyt de Provincie van Hollandt "
        "ende Welt-Vrieslandt. Velters, Ockere , Noey; van Hoorn , met een extraordinaris "
        "Gedeputeerde uyt de Provincie van Zeelandt. Van Renswoude , van Voor{t. Van "
        "Schwartzenbergh, vander Waayen, Vegilin Van I{elmuden. Van Iddekinge ‚ van Tamminga.")


```

The `find_matches` method returns match objects:


```python
# look for matches in the first example text
matches = fuzzy_searcher.find_matches(text)
```


Printing the matches directly yields the following output:


```python
for match in matches:
    print(match)

```

    PhraseMatch(phrase: "Veneris", variant: "Veneris", string: "Veucris", offset: 3, ignorecase: False, levenshtein_similarity: 0.7142857142857143)
    PhraseMatch(phrase: "den .. Januarii 1725", variant: "den .. Januarii 1725", string: "den 5. Januaris 1725", offset: 11, ignorecase: False, levenshtein_similarity: 0.9)
    PhraseMatch(phrase: "PRAESIDE", variant: "PRAESIDE", string: "PR&ASIDE", offset: 33, ignorecase: False, levenshtein_similarity: 0.75)
    PhraseMatch(phrase: "PRAESENTIBUS", variant: "PRAESENTIBUS", string: "PRASENTIEBUS", offset: 63, ignorecase: False, levenshtein_similarity: 0.8333333333333334)


When you have many texts, you can pass them as simple dictionaries with a `text` property and an `id` property. The `id` will be added to the matches, so you can keep track of which match comes from which text.


```python
texts = [
    {
        'text': ("ie Veucris den 5. Januaris 1725. PR&ASIDE, Den Heere Bentinck. PRASENTIEBUS, "
                 "De Heeren Jan Welderen , van Dam, Torck , met een extraordinaris Gedeputeerde "
                 "uyt de Provincie van Gelderlandt. Van Maasdam , vanden Boeizelaar , "
                 "Raadtpenfionaris van Hoornbeeck , met een extraordinaris Gedeputeerde uyt de "
                 "Provincie van Hollandt ende Welt-Vrieslandt. Velters, Ockere , Noey; van Hoorn , "
                 "met een extraordinaris Gedeputeerde uyt de Provincie van Zeelandt. Van Renswoude "
                 ", van Voor{t. Van Schwartzenbergh, vander Waayen, Vegilin Van I{elmuden. Van "
                 "Iddekinge ‚ van Tamminga."),
        'id': 'session-3780-num-3-para-1'
    },
    {
        'text': ("Mercuri: den 10. Jangarii, 1725. ia PRESIDE, Den Heere an Iddekinge. PRA&SENTIBUS, "
                 "De Heeren /an Welderen , van Dam, van Wynbergen, Torck, met een extraordinaris "
                 "Gedeputeerde uyt de Provincie van Gelderland. Van Maasdam , Raadtpenfionaris van "
                 "Hoorn=beeck. Velters, Ockerfe, Noey. Taats van Amerongen, van Renswoude. Vander "
                 "Waasen , Vegilin, ’ Bentinck, van I(elmaden. Van Tamminga."),
        'id': 'session-3780-num-7-para-1'
    }
]

# look for matches in a list of texts
for text in texts:
    for match in fuzzy_searcher.find_matches(text):
        # the phrase objects from the phrase model are added as a property
        # so you have easy access to it, to get e.g. the phrase_string
        print(f"{match.phrase.phrase_string: <20}\t{match.string: <20}"
              f"\t{match.offset: >5}\t{match.levenshtein_similarity: >.2f}\t{match.text_id}")

```

    Veneris             	Veucris             	    3	0.71	session-3780-num-3-para-1
    den .. Januarii 1725	den 5. Januaris 1725	   11	0.90	session-3780-num-3-para-1
    PRAESIDE            	PR&ASIDE            	   33	0.75	session-3780-num-3-para-1
    PRAESENTIBUS        	PRASENTIEBUS        	   63	0.83	session-3780-num-3-para-1
    Mercurii            	Mercuri             	    0	0.88	session-3780-num-7-para-1
    den .. Januarii 1725	den 10. Jangarii, 1725	    9	0.82	session-3780-num-7-para-1
    PRAESIDE            	PRESIDE             	   36	0.88	session-3780-num-7-para-1
    PRAESENTIBUS        	PRA&SENTIBUS        	   69	0.92	session-3780-num-7-para-1


Phrases can be strings, but can also represented as dictionaries with extra information. For instance, it's possible to add labels and metadata.


```python
domain_phrases = [
    # terms for the chair and attendants of a meeting
    {
        'phrase': 'PRAESIDE',
        'label': 'president',
        'metadata': {
            'lang': 'latin'
        }
    },
    {
        'phrase': 'PRAESENTIBUS',
        'label': 'attendants',
        'metadata': {
            'lang': 'latin'
        }
    },
    # some weekdays in Latin
    {
        'phrase': 'Veneris',
        'label': ['date', 'weekday'],
        'metadata': {
            'lang': 'latin'
        }
    },
    {
        'phrase': 'Mercurii',
        'label': ['date', 'weekday'],
        'metadata': {
            'lang': 'latin'
        }
    },
    # some date phrase where any date in January 1725 should match
    {
        'phrase': 'den .. Januarii 1725',
        'label': ['date', 'full_date'],
        'metadata': {
            'lang': 'dutch'
        }
    }    
]

# initialize a new searcher instance with the phrases and the config
fuzzy_searcher = FuzzyPhraseSearcher(config=config, phrase_model=domain_phrases)

# look for matches in the first example text
matches = fuzzy_searcher.find_matches(texts[0])

[match.phrase.metadata['lang'] for match in matches]
```




    ['latin', 'dutch', 'latin', 'latin']



This makes it easy to filter matches on metadata properties:


```python
[match for match in matches if match.phrase.metadata['lang'] == 'dutch']
```




    [PhraseMatch(phrase: "den .. Januarii 1725", variant: "den .. Januarii 1725", string: "den 5. Januaris 1725", offset: 11, ignorecase: False, levenshtein_similarity: 0.9)]



## Phrase Variants and Distractors

If you have multiple phrases representing the same meaning or function, it is possible to register them in a single Phrase object and ask the fuzzy searcher to match with any of the variants.

By default, the fuzzy searcher ignores variants for efficiency. To make sure the searcher uses them, set the `include_variants` property in the config to `True`:


```python
attendants_phrases = [
    {
        'phrase': 'PRAESENTIBUS',
        'variants': [
            'Present de Heeren',
            'Pntes die voors'
        ],
        'label': 'attendants'
    }
]

config = {
    "char_match_threshold": 0.6,
    "ngram_threshold": 0.5,
    "levenshtein_threshold": 0.6,
    "ignorecase": False,
    "include_variants": True,
    "max_length_variance": 3,
    "ngram_size": 2,
    "skip_size": 2,
}


# initialize a new searcher instance with the phrases and the config
fuzzy_searcher = FuzzyPhraseSearcher(config=config, phrase_model=attendants_phrases)

texts = [
    {
        'text': 'ie Veucris den 5. Januaris 1725. PR&ASIDE, Den Heere Bentinck. PRASENTIEBUS, De Heeren Jan Welderen , van Dam, Torck , met een extraordinaris Gedeputeerde uyt de Provincie van Gelderlandt. Van Maasdam , vanden Boeizelaar , Raadtpenfionaris van Hoornbeeck , met een extraordinaris Gedeputeerde uyt de Provincie van Hollandt ende Welt-Vrieslandt. Velters, Ockere , Noey; van Hoorn , met een extraordinaris Gedeputeerde uyt de Provincie van Zeelandt. Van Renswoude , van Voor{t. Van Schwartzenbergh, vander Waayen, Vegilin Van I{elmuden. Van Iddekinge ‚ van Tamminga.',
        'id': 'session-3780-num-3-para-1'
    },
    {
        'text': 'Mercuri: den 10. Jangarii, 1725. ia PRESIDE, Den Heere an Iddekinge. PRA&SENTIBUS, De Heeren /an Welderen , van Dam, van Wynbergen, Torck, met een extraordinaris Gedeputeerde uyt de Provincie van Gelderland. Van Maasdam , Raadtpenfionaris van Hoorn=beeck. Velters, Ockerfe, Noey. Taats van Amerongen, van Renswoude. Vander Waasen , Vegilin, ’ Bentinck, van I(elmaden. Van Tamminga.',
        'id': 'session-3780-num-7-para-1'
    },
    {
        'text': 'Praeside de Heer de Knuijt, Praseat de Heeron Verbolt, Raesfelt, Huijgens, Ommercn, Wimmenum, Loo, Cats, Rhijnhuijsen, van der Hoolck Jrovestins, Eissinge,',
        'id': 'session-3210-num-166-para-1'
    }
]

for text in texts:
    matches = fuzzy_searcher.find_matches(text)
    for match in matches:
        print(f"{match.phrase.phrase_string: <20}\t{match.string: <20}"
              f"\t{match.offset: >5}\t{match.levenshtein_similarity: >.2f}\t{match.text_id}")
```

    PRAESENTIBUS        	PRASENTIEBUS        	   63	0.83	session-3780-num-3-para-1
    PRAESENTIBUS        	PRA&SENTIBUS        	   69	0.92	session-3780-num-7-para-1
    PRAESENTIBUS        	Praeside de Heer    	    0	0.65	session-3210-num-166-para-1
    PRAESENTIBUS        	Praseat de Heeron   	   28	0.82	session-3210-num-166-para-1


The  phrase "Praeside de Heer" is a formulaic phrase to signal the president instead of the attendants, but also matches with the PRAESENTIBUS phrase. 

Since it's formulaic and is frequently used in combination with the formulaic phrase "Present de Heeren", it will lead to many unwanted matches. To avoid them, it is possible to add it as a `distractor` phrase. Distractors are orthographically similar phrases that have a different meaning or function.


```python
attendants_phrases = [
    {
        'phrase': 'PRAESENTIBUS',
        'variants': [
            'Present de Heeren',
            'Pntes die voors'
        ],
        'distractors': [
            'Praeside de Heer'
        ],
        'label': 'attendants'
    }
]

config = {
    "char_match_threshold": 0.6,
    "ngram_threshold": 0.5,
    "levenshtein_threshold": 0.6,
    "ignorecase": False,
    "include_variants": True,
    "filter_distractors": True,
    "max_length_variance": 3,
    "ngram_size": 2,
    "skip_size": 2,
}


# initialize a new searcher instance with the phrases and the config
fuzzy_searcher = FuzzyPhraseSearcher(config=config, phrase_model=attendants_phrases)

for text in texts:
    matches = fuzzy_searcher.find_matches(text)
    for match in matches:
        print(f"{match.phrase.phrase_string: <20}\t{match.string: <20}"
              f"\t{match.offset: >5}\t{match.levenshtein_similarity: >.2f}\t{match.text_id}")
```

    PRAESENTIBUS        	PRASENTIEBUS        	   63	0.83	session-3780-num-3-para-1
    PRAESENTIBUS        	PRA&SENTIBUS        	   69	0.92	session-3780-num-7-para-1
    PRAESENTIBUS        	Praseat de Heeron   	   28	0.82	session-3210-num-166-para-1


The formulaic phrase _'Praeside de Heer'_ is a variant of the phrase _'PRAESENTIBUS'_. If it is registered as a variant, the fuzzy searcher will match occurrences like "Praseat de Heeron" with both _'Praeside de Heer'_ and _'Present de Heeren'_, but with both matches representing the same text string, the searcher will pick the best matching one. 


```python
attendants_phrases = [
    {
        'phrase': 'PRAESENTIBUS',
        'variants': [
            'Present de Heeren',
            'Pntes die voors'
        ],
        'label': 'attendants'
    },
    {
        'phrase': 'PRAESIDE',
        'variants': [
            'Praeside de Heer',
        ],
        'label': 'president'
    }
]

config = {
    "char_match_threshold": 0.6,
    "ngram_threshold": 0.5,
    "levenshtein_threshold": 0.6,
    "ignorecase": False,
    "include_variants": True,
    "filter_distractors": True,
    "max_length_variance": 3,
    "ngram_size": 2,
    "skip_size": 2,
}


# initialize a new searcher instance with the phrases and the config
fuzzy_searcher = FuzzyPhraseSearcher(config=config, phrase_model=attendants_phrases)

for text in texts:
    matches = fuzzy_searcher.find_matches(text)
    for match in matches:
        print(f"{match.phrase.phrase_string: <20}\t{match.string: <20}"
              f"\t{match.offset: >5}\t{match.levenshtein_similarity: >.2f}\t{match.text_id}")
```

    PRAESIDE            	PR&ASIDE            	   33	0.75	session-3780-num-3-para-1
    PRAESENTIBUS        	PRASENTIEBUS        	   63	0.83	session-3780-num-3-para-1
    PRAESIDE            	PRESIDE             	   36	0.88	session-3780-num-7-para-1
    PRAESENTIBUS        	PRA&SENTIBUS        	   69	0.92	session-3780-num-7-para-1
    PRAESIDE            	Praeside de Heer    	    0	1.00	session-3210-num-166-para-1
    PRAESENTIBUS        	Praseat de Heeron   	   28	0.82	session-3210-num-166-para-1


## Alternative Representations

Alternatively, each match object can generate a JSON representation of the match containing all information:



```python
import json

matches =[match for text in texts for match in fuzzy_searcher.find_matches(text)]

for match in matches:
    print(json.dumps(match.json(), indent=4))

```

    {
        "type": "PhraseMatch",
        "phrase": "PRAESIDE",
        "variant": "PRAESIDE",
        "string": "PR&ASIDE",
        "offset": 33,
        "label": "president",
        "ignorecase": false,
        "text_id": "session-3780-num-3-para-1",
        "match_scores": {
            "char_match": 0.875,
            "ngram_match": 0.6666666666666666,
            "levenshtein_similarity": 0.75
        }
    }
    {
        "type": "PhraseMatch",
        "phrase": "PRAESENTIBUS",
        "variant": "PRAESENTIBUS",
        "string": "PRASENTIEBUS",
        "offset": 63,
        "label": "attendants",
        "ignorecase": false,
        "text_id": "session-3780-num-3-para-1",
        "match_scores": {
            "char_match": 1.0,
            "ngram_match": 0.7692307692307693,
            "levenshtein_similarity": 0.8333333333333334
        }
    }
    {
        "type": "PhraseMatch",
        "phrase": "PRAESIDE",
        "variant": "PRAESIDE",
        "string": "PRESIDE",
        "offset": 36,
        "label": "president",
        "ignorecase": false,
        "text_id": "session-3780-num-7-para-1",
        "match_scores": {
            "char_match": 0.875,
            "ngram_match": 0.7777777777777778,
            "levenshtein_similarity": 0.875
        }
    }
    {
        "type": "PhraseMatch",
        "phrase": "PRAESENTIBUS",
        "variant": "PRAESENTIBUS",
        "string": "PRA&SENTIBUS",
        "offset": 69,
        "label": "attendants",
        "ignorecase": false,
        "text_id": "session-3780-num-7-para-1",
        "match_scores": {
            "char_match": 0.9166666666666666,
            "ngram_match": 0.8461538461538461,
            "levenshtein_similarity": 0.9166666666666666
        }
    }
    {
        "type": "PhraseMatch",
        "phrase": "PRAESIDE",
        "variant": "Praeside de Heer",
        "string": "Praeside de Heer",
        "offset": 0,
        "label": "president",
        "ignorecase": false,
        "text_id": "session-3210-num-166-para-1",
        "match_scores": {
            "char_match": 1.0,
            "ngram_match": 1.0,
            "levenshtein_similarity": 1.0
        }
    }
    {
        "type": "PhraseMatch",
        "phrase": "PRAESENTIBUS",
        "variant": "Present de Heeren",
        "string": "Praseat de Heeron",
        "offset": 28,
        "label": "attendants",
        "ignorecase": false,
        "text_id": "session-3210-num-166-para-1",
        "match_scores": {
            "char_match": 0.8235294117647058,
            "ngram_match": 0.6666666666666666,
            "levenshtein_similarity": 0.8235294117647058
        }
    }


The `variant` property shows which phrase (main phrase or variant) was used for matching.

### Web Annotations

Match objects can also generate Web Annotation representations:


```python
matches = fuzzy_searcher.find_matches(texts[0])

import json

print(json.dumps(matches[0].as_web_anno(), indent=2))
```

    {
      "@context": "http://www.w3.org/ns/anno.jsonld",
      "id": "cbf33232-a07a-4f2f-8015-942ec000452b",
      "type": "Annotation",
      "motivation": "classifying",
      "created": "2023-12-12T17:08:55.001999",
      "generator": {
        "id": "https://github.com/marijnkoolen/fuzzy-search",
        "type": "Software",
        "name": "fuzzy-search v2.0.1a"
      },
      "target": {
        "source": "session-3780-num-3-para-1",
        "selector": {
          "type": "TextPositionSelector",
          "start": 33,
          "end": 41
        }
      },
      "body": [
        {
          "type": "TextualBody",
          "purpose": "tagging",
          "format": "text",
          "value": "PRAESIDE"
        },
        {
          "type": "TextualBody",
          "purpose": "highlighting",
          "format": "text",
          "value": "PR&ASIDE"
        },
        {
          "type": "TextualBody",
          "purpose": "correcting",
          "format": "text",
          "value": "PRAESIDE"
        },
        {
          "type": "TextualBody",
          "purpose": "classifying",
          "format": "text",
          "value": "president"
        }
      ]
    }


## Configuration

The key configurable properties are:

- the size of character ngrams (`ngram_size`, default is 2)
- the number of skips (`skip_size`, default is 2)

Skip-grams with these settings lead to a very exhaustive search.

Other configurable properties:

- `max_length_variance`: the maximum number of characters that a phrase and a matching occurrence in the text can vary in length.
- `include_variants`: Boolean to ignore or include phrase variants.
- `filter_distractors`: Boolean to ignore or include distractor phrases to filter out matches that are closer to distractors than to the registered phrases and variants.

### The Impact of Ngram Size and Skipgram Size

As mentioned above, the default setting leads to very exhaustive search, but will slow down search when the number of phrases to search for gets big (a few dozen is enough to make it _very_ slow). This is due to the nature of short character ngrams (two-character strings contain little information and are more likely to occur in words than longer ngrams) and the number of _distinct_ ngrams that are created by introducing one or more skips (two-character sequences are constrained in each language, but when skips are introduced, it creates combinations that do not occur naturally in text).

This is demonstrated below.


```python
from fuzzy_search.tokenization.string import text2skipgrams

sent = "This is a test sentence."

skipgrams = [skipgram for skipgram in text2skipgrams(sent)]
print([s.string for s in skipgrams])
print('number of skipgrams:', len(skipgrams))
```

    ['Th', 'Ti', 'Ts', 'hi', 'hs', 'h ', 'is', 'i ', 'ii', 's ', 'si', 'ss', ' i', ' s', '  ', 'is', 'i ', 'ia', 's ', 'sa', 's ', ' a', '  ', ' t', 'a ', 'at', 'ae', ' t', ' e', ' s', 'te', 'ts', 'tt', 'es', 'et', 'e ', 'st', 's ', 'ss', 't ', 'ts', 'te', ' s', ' e', ' n', 'se', 'sn', 'st', 'en', 'et', 'ee', 'nt', 'ne', 'nn', 'te', 'tn', 'tc', 'en', 'ec', 'ee', 'nc', 'ne', 'n.', 'ce', 'c.', 'e.']
    number of skipgrams: 66


Increasing the ngram size strongly reduces the number of matches between phrases and the target text (character 3 grams are more specific than 2 grams), while increasing the skip size strongly increasing the number of skipgrams (allowing for 2 skips creates many more permutations than allowing for 1 skip).

To see the impact of different skip sizes, count the number of skipgrams for the sentence above for skip sizes of 2, 1 and 0 respectively:


```python
skipgrams = [skipgram for skipgram in text2skipgrams(sent)]
print('number of skipgrams:', len(skipgrams))
skipgrams = [skipgram for skipgram in text2skipgrams(sent, ngram_size=2, skip_size=1)]
print('number of skipgrams:', len(skipgrams))
skipgrams = [skipgram for skipgram in text2skipgrams(sent, ngram_size=2, skip_size=0)]
print('number of skipgrams:', len(skipgrams))

```

    number of skipgrams: 66
    number of skipgrams: 45
    number of skipgrams: 23


The same for using `ngram_size=3`:


```python
skipgrams = [skipgram for skipgram in text2skipgrams(sent, ngram_size=3, skip_size=2)]
print('number of skipgrams:', len(skipgrams))
skipgrams = [skipgram for skipgram in text2skipgrams(sent, ngram_size=3, skip_size=1)]
print('number of skipgrams:', len(skipgrams))
skipgrams = [skipgram for skipgram in text2skipgrams(sent, ngram_size=3, skip_size=0)]
print('number of skipgrams:', len(skipgrams))

```

    number of skipgrams: 124
    number of skipgrams: 64
    number of skipgrams: 22


## The Fuzzy Token Searcher 

### Using Word Tokens Instead of Character Ngrams

With the default settings for `ngram_size` and `skip_size`, the searcher becomes very slow when the phrase model contains many phrases. An alternative to increasing the ngram size and decreasing the skip size is to use the `FuzzyTokenSearcher`. It turns the phrases and the target text into lists of word tokens (the tokenizer is configurable) and uses character skip grams to identify candidate phrase tokens matching tokens in the text. It then uses token sequences to identify fuzzy matches. 

This speeds up the search (especially for the default settings `ngram_size=2` and `skip_size=2`) at the cost of slightly less exhaustive search. On test corpus of 1000 texts (1 million words) with relatively high HTR quality (CER around 5%), and with a model with 67 phrases, the `FuzzyTokenSearcher` finds the same number of matches as the `FuzzyPhraseSearcher`, but is 70-100 times faster. 


```python
from fuzzy_search import FuzzyTokenSearcher

# initialize a new searcher instance with the phrases and the config
fuzzy_searcher = FuzzyTokenSearcher(config=config, phrase_model=domain_phrases)

# take some example texts: resolutoins of the meeting of the Dutch States General on
# Friday the 5th of January, 1725
text = ("ie Veucris den 5. Januaris 1725. PR&ASIDE, Den Heere Bentinck. PRASENTIEBUS, "
        "De Heeren Jan Welderen , van Dam, Torck , met een extraordinaris Gedeputeerde "
        "uyt de Provincie van Gelderlandt. Van Maasdam , vanden Boeizelaar , Raadtpenfionaris "
        "van Hoornbeeck , met een extraordinaris Gedeputeerde uyt de Provincie van Hollandt "
        "ende Welt-Vrieslandt. Velters, Ockere , Noey; van Hoorn , met een extraordinaris "
        "Gedeputeerde uyt de Provincie van Zeelandt. Van Renswoude , van Voor{t. Van "
        "Schwartzenbergh, vander Waayen, Vegilin Van I{elmuden. Van Iddekinge ‚ van Tamminga.")

fuzzy_searcher.find_matches(text)
```




    [PhraseMatch(phrase: "Veneris", variant: "Veneris", string: "Veucris", offset: 3, ignorecase: False, levenshtein_similarity: 0.7142857142857143),
     PhraseMatch(phrase: "PRAESIDE", variant: "PRAESIDE", string: "PR ASIDE", offset: 33, ignorecase: False, levenshtein_similarity: 0.75),
     PhraseMatch(phrase: "PRAESENTIBUS", variant: "PRAESENTIBUS", string: "PRASENTIEBUS", offset: 63, ignorecase: False, levenshtein_similarity: 0.8333333333333334)]




```python

```


```python

```


```python

```


```python

```
