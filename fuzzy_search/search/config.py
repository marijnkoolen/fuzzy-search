import string

default_config = {
    # these thresholds work when there are quite a few OCR errors
    # use higher thresholds for higher quality OCR to avoid
    # spurious matches.
    "char_match_threshold": 0.6,
    "ngram_threshold": 0.5,
    "levenshtein_threshold": 0.6,
    "skipgram_threshold": 0.2,
    # Is upper/lowercase a meaningful signal?
    "ignorecase": False,
    # should matches follow word boundaries?
    "use_word_boundaries": False,
    # for phrases that have variant phrasings
    "include_variants": False,
    # avoid matching with similar but different phrases
    "filter_distractors": False,
    # matching string can be lower/shorter than prhase
    "max_length_variance": 1,
    # higher ngram size allows fewer character differences
    "ngram_size": 2,
    # fewer skips is much faster but less exhaustive
    "skip_size": 2,
    # first check for exact matches to speed up fuzzy search
    "skip_exact_matching": False,
    # allow matches of partially overlapping phrase
    "allow_overlapping_matches": False,
    # the set of symbols to use as punctuation (for word boundaries)
    "punctuation": string.punctuation,
    "debug": False
}