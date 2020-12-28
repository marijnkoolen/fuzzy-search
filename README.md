# fuzzy-search
Fuzzy search module for searching lists of words in low quality OCR and HTR text.

Project page on PyPI: [https://pypi.org/project/fuzzy-search/](https://pypi.org/project/fuzzy-search/)

## Installing

```commandline
pip install -u fuzzy-search
```

## Usage

```python
from fuzzy_search.fuzzy_phrase_searcher import FuzzyPhraseSearcher
from fuzzy_search.fuzzy_phrase_model import PhraseModel

# highger matching thresholds for higher quality OCR/HTR (higher precision, recall should be good anyway)
# lower matching thresholds for lower quality OCR/HTR (higher recall, as that's the main problem)
config = {
    "char_match_threshold": 0.8,
    "ngram_threshold": 0.6,
    "levenshtein_threshold": 0.8,
    "ignorecase": False,
    "ngram_size": 3,
    "skip_size": 0,
}

# initialize a new searcher instance with the config
fuzzy_searcher = FuzzyPhraseSearcher(config)

# create a list of domain keywords and phrases
domain_phrases = [
    # terms for the chair and attendants of a meeting
    "PRAESIDE",
    "PRAESENTIBUS",
    # some weekdays in Latin
    "Veneris", 
    "Mercuri",
    # some date phrase where any date in January 1725 should match
    "den .. Januarii 1725"
]

# create a PhraseModel object from the domain phrases
phrase_model = PhraseModel(phrases=domain_phrases)

# register the phrase model with the searcher
fuzzy_searcher.index_phrase_model(phrase_model)

# take some example texts: meetings of the Dutch States General in January 1725
text1 = "ie Veucris den 5. Januaris 1725. PR&ASIDE, Den Heere Bentinck. PRASENTIEBUS, De Heeren Jan Welderen , van Dam, Torck , met een extraordinaris Gedeputeerde uyt de Provincie van Gelderlandt. Van Maasdam , vanden Boeizelaar , Raadtpenfionaris van Hoornbeeck , met een extraordinaris Gedeputeerde uyt de Provincie van Hollandt ende Welt-Vrieslandt. Velters, Ockere , Noey; van Hoorn , met een extraordinaris Gedeputeerde uyt de Provincie van Zeelandt. Van Renswoude , van Voor{t. Van Schwartzenbergh, vander Waayen, Vegilin Van I{elmuden. Van Iddekinge ‚ van Tamminga."

text2 = "Mercuri: den 10. Jangarii , | 1725. ia PRESIDE, Den Heere an Iddekinge. PRA&SENTIBUS, De Heeren /an Welderen , van Dam, van Wynbergen, Torck, met een extraordinaris Gedeputeerde uyt de Provincie van Gelderland. Van Maasdam , Raadtpenfionaris van Hoorn=beeck. Velters, Ockerfe, Noey. Taats van Amerongen, van Renswoude. Vander Waasen , Vegilin, ’ Bentinck, van I(elmaden. Van Tamminga."

```

The `find_matches` method returns match objects:

```python
# look for matches in the first example text
for match in fuzzy_searcher.find_matches(text1):
    print(match)
```

Printing the matches directly yields the following output:
```python
Match(phrase: "Veneris", variant: "Veneris",string: "Veucris", offset: 3)
Match(phrase: "den .. Januarii 1725", variant: "den .. Januarii 1725",string: "den 5. Januaris 1725.", offset: 11)
Match(phrase: "PRAESIDE", variant: "PRAESIDE",string: "PR&ASIDE,", offset: 33)
Match(phrase: "PRAESENTIBUS", variant: "PRAESENTIBUS",string: "PRASENTIEBUS,", offset: 63)
```

Alternatively, each match object can generate a JSON representation of the match containing all information:

```python
# look for matches in the first example text
for match in fuzzy_searcher.find_matches(text1):
    print(match.json())
```

This yields more detailed output:

```js
{'match_keyword': 'Veneris', 'match_term': 'Veneris', 'match_string': 'Veucris', 'match_offset': 3, 'char_match': 0.7142857142857143, 'ngram_match': 0.625, 'levenshtein_distance': 0.7142857142857143}
{'match_keyword': 'den .. Januarii 1725', 'match_term': 'den .. Januarii 1725', 'match_string': 'den 5. Januaris 1725', 'match_offset': 11, 'char_match': 0.9, 'ngram_match': 0.8095238095238095, 'levenshtein_distance': 0.9}
{'match_keyword': 'PRAESIDE', 'match_term': 'PRAESIDE', 'match_string': 'PR&ASIDE', 'match_offset': 33, 'char_match': 0.875, 'ngram_match': 0.6666666666666666, 'levenshtein_distance': 0.75}
{'match_keyword': 'PRAESENTIBUS', 'match_term': 'PRAESENTIBUS', 'match_string': 'PRASENTIEBUS', 'match_offset': 63, 'char_match': 1.0, 'ngram_match': 0.7692307692307693, 'levenshtein_distance': 0.8333333333333334}
```

Running the searcher on the second text:

```python
# look for matches in the second example text
for match in fuzzy_searcher.find_candidates(text2):
    print(match.json())
```

This yields the following output:

```js
{'phrase': 'Veneris', 'variant': 'Veneris', 'string': 'Veucris', 'offset': 3, 'match_scores': {'char_match': 0.7142857142857143, 'ngram_match': 0.625, 'levenshtein_similarity': 0.7142857142857143}}
{'phrase': 'den .. Januarii 1725', 'variant': 'den .. Januarii 1725', 'string': 'den 5. Januaris 1725.', 'offset': 11, 'match_scores': {'char_match': 0.95, 'ngram_match': 0.7619047619047619, 'levenshtein_similarity': 0.8571428571428572}}
{'phrase': 'PRAESIDE', 'variant': 'PRAESIDE', 'string': 'PR&ASIDE,', 'offset': 33, 'match_scores': {'char_match': 0.875, 'ngram_match': 0.5555555555555556, 'levenshtein_similarity': 0.6666666666666667}}
{'phrase': 'PRAESENTIBUS', 'variant': 'PRAESENTIBUS', 'string': 'PRASENTIEBUS,', 'offset': 63, 'match_scores': {'char_match': 1.0, 'ngram_match': 0.6923076923076923, 'levenshtein_similarity': 0.7692307692307692}}
```

## Matches as Web Annotations

If texts are passed to `find_matches` as dictionaries with an identifier, the resulting matches
include the text identifier and can generate Web Annotation representations:

```python
# create a dictionary for the second text and add an identifier
text2_with_id = {
    "text": text2,
    "id": "urn:republic:3783_0076:page=151:para=4"
}
matches = fuzzy_searcher.find_matches(text2_with_id)

import json

# use json.dumps to pretty print the first match as Web Annotation
print(json.dumps(matches[0].as_web_anno(), indent=2))
```

Output:

```json
{
  "@context": "http://www.w3.org/ns/anno.jsonld",
  "id": "cca6740d-e584-4322-b517-67d92e0e508a",
  "type": "Annotation",
  "motivation": "classifying",
  "created": "2020-12-08T10:22:26.838154",
  "generator": {
    "id": "https://github.com/marijnkoolen/fuzzy-search",
    "type": "Software",
    "name": "FuzzySearcher"
  },
  "target": {
    "source": "urn:republic:3783_0076:page=151:para=4",
    "selector": {
      "type": "TextPositionSelector",
      "start": 0,
      "end": 8
    }
  },
  "body": {
    "type": "Dataset",
    "value": {
      "match_phrase": "Mercurii",
      "match_variant": "Mercurii",
      "match_string": "Mercuri:",
      "phrase_metadata": {
        "phrase": "Mercurii"
      }
    }
  }
}
```



## Documentation To Do

- adding variant phrases and distractors
- multiple searchers and searching in the context of other matches
