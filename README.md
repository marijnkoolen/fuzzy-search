# fuzzy-search
Fuzzy search modules for searching lists of words in low quality OCR and HTR text.

## Usage

```python
from fuzzy_search.fuzzy_keyword_searcher import FuzzyKeywordSearcher

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
fuzzy_searcher = FuzzyContextSearcher(config)

i# create a list of domain keywords and phrases
keywords = [
    # terms for the chair and attendants of a meeting
    "PRAESIDE",
    "PRAESENTIBUS",
    # some weekdays in Latin
    "Veneris", 
    "Mercuri",
    # some date phrase where any date in January 1725 should match
    "den .. Januarii 1725"
]

# register the keywords with the searcher
fuzzy_searcher.index_keywords(domain_keywords)

# take some example texts: meetings of the Dutch States General in January 1725
text1 = "ie Veucris den 5. Januaris 1725. PR&ASIDE, Den Heere Bentinck. PRASENTIEBUS, De Heeren Jan Welderen , van Dam, Torck , met een extraordinaris Gedeputeerde uyt de Provincie van Gelderlandt. Van Maasdam , vanden Boeizelaar , Raadtpenfionaris van Hoornbeeck , met een extraordinaris Gedeputeerde uyt de Provincie van Hollandt ende Welt-Vrieslandt. Velters, Ockere , Noey; van Hoorn , met een extraordinaris Gedeputeerde uyt de Provincie van Zeelandt. Van Renswoude , van Voor{t. Van Schwartzenbergh, vander Waayen, Vegilin Van I{elmuden. Van Iddekinge ‚ van Tamminga."

text2 = "Mercuri: den 10. Jangarii , | 1725. ia PRESIDE, Den Heere an Iddekinge. PRA&SENTIBUS, De Heeren /an Welderen , van Dam, van Wynbergen, Torck, met een extraordinaris Gedeputeerde uyt de Provincie van Gelderland. Van Maasdam , Raadtpenfionaris van Hoorn=beeck. Velters, Ockerfe, Noey. Taats van Amerongen, van Renswoude. Vander Waasen , Vegilin, ’ Bentinck, van I(elmaden. Van Tamminga."

```

The `find_candidates` method yields match objects:

```python
# look for matches in the first example text
for match in fuzzy_searcher.find_candidates(text1):
    print(match)
```

```json
{'match_keyword': 'Veneris', 'match_term': 'Veneris', 'match_string': 'Veucris', 'match_offset': 3, 'char_match': 0.7142857142857143, 'ngram_match': 0.625, 'levenshtein_distance': 0.7142857142857143}
{'match_keyword': 'den .. Januarii 1725', 'match_term': 'den .. Januarii 1725', 'match_string': 'den 5. Januaris 1725', 'match_offset': 11, 'char_match': 0.9, 'ngram_match': 0.8095238095238095, 'levenshtein_distance': 0.9}
{'match_keyword': 'PRAESIDE', 'match_term': 'PRAESIDE', 'match_string': 'PR&ASIDE', 'match_offset': 33, 'char_match': 0.875, 'ngram_match': 0.6666666666666666, 'levenshtein_distance': 0.75}
{'match_keyword': 'PRAESENTIBUS', 'match_term': 'PRAESENTIBUS', 'match_string': 'PRASENTIEBUS', 'match_offset': 63, 'char_match': 1.0, 'ngram_match': 0.7692307692307693, 'levenshtein_distance': 0.8333333333333334}
```

```python
i# look for matches in the second example text
for match in fuzzy_searcher.find_candidates(text2):
    print(match)
```


```json
{'match_keyword': 'Mercuri', 'match_term': 'Mercuri', 'match_string': 'Mercuri', 'match_offset': 0, 'char_match': 1.0, 'ngram_match': 1.0, 'levenshtein_distance': 1.0}
{'match_keyword': 'den .. Januarii 1725', 'match_term': 'den .. Januarii 1725', 'match_string': 'den 10. Jangarii, 1725', 'match_offset': 9, 'char_match': 0.9, 'ngram_match': 0.7619047619047619, 'levenshtein_distance': 0.8181818181818181}
{'match_keyword': 'PRAESIDE', 'match_term': 'PRAESIDE', 'match_string': 'PRESIDE', 'match_offset': 36, 'char_match': 0.875, 'ngram_match': 0.7777777777777778, 'levenshtein_distance': 0.875}
{'match_keyword': 'PRAESENTIBUS', 'match_term': 'PRAESENTIBUS', 'match_string': 'PRA&SENTIBUS', 'match_offset': 69, 'char_match': 0.9166666666666666, 'ngram_match': 0.8461538461538461, 'levenshtein_distance': 0.9166666666666666}
```
