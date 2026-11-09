"""Check that FuzzyPhraseSearcher still returns the matches stored in the regression fixture."""
import gzip
import json
import os
from pathlib import Path

import pytest

from fuzzy_search.search.phrase_searcher import FuzzyPhraseSearcher

FIXTURE_DIR = Path(os.environ.get('FUZZY_REGRESSION_DIR', Path(__file__).parent / 'fixtures'))
FIXTURE_NAME = os.environ.get('FUZZY_REGRESSION_NAME', 'phrase_regression')
META_FILE = FIXTURE_DIR / f'{FIXTURE_NAME}.meta.json'
CASES_FILE = FIXTURE_DIR / f'{FIXTURE_NAME}.jsonl.gz'
SCORE_FIELDS = ('char_match', 'ngram_match', 'levenshtein_similarity')
# The quick run checks every QUICK_STRIDE-th case; pass --regression-full to check them all.
QUICK_STRIDE = 20

pytestmark = pytest.mark.skipif(not META_FILE.exists(), reason='regression fixture has not been built')


def load_metadata():
    with open(META_FILE, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


def load_cases():
    with gzip.open(CASES_FILE, 'rt', encoding='utf-8') as fh:
        return [json.loads(line) for line in fh]


def comparable(match):
    return {field: value for field, value in match.items() if field not in SCORE_FIELDS}


def actual_matches(searcher, text):
    matches = [{'phrase': m.phrase.phrase_string, 'variant': m.variant.phrase_string, 'string': m.string,
                'offset': m.offset, 'end': m.end, 'label': m.label, 'char_match': m.character_overlap,
                'ngram_match': m.ngram_overlap, 'levenshtein_similarity': m.levenshtein_similarity}
               for m in searcher.find_matches(text)]
    return sorted(matches, key=lambda m: (m['offset'], m['end'], m['phrase'], m['variant'], m['string']))


def match_key(match):
    return tuple(match[field] for field in ('phrase', 'variant', 'string', 'offset', 'end', 'label'))


def compare(actual, expected):
    """Return (missing, changed, extra) between actual and expected matches of one text.

    missing: expected matches not found; changed: found but with different scores; extra: new matches."""
    actual_by_key = {match_key(m): m for m in actual}
    expected_keys = {match_key(m) for m in expected}
    missing, changed = [], []
    for e in expected:
        a = actual_by_key.get(match_key(e))
        if a is None:
            missing.append(e)
        elif any(a[f] != pytest.approx(e[f], abs=1e-9) for f in SCORE_FIELDS):
            changed.append({'expected': e, 'actual': a})
    extra = [a for k, a in actual_by_key.items() if k not in expected_keys]
    return missing, changed, extra


@pytest.mark.parametrize('key', sorted(load_metadata()['configs']) if META_FILE.exists() else [])
def test_matches_cover_fixture(key, request):
    """Every stored match must still be found with the same scores. New matches are allowed (the stored
    set is a lower bound) but are reported; --regression-exact also fails on them."""
    metadata = load_metadata()
    searcher = FuzzyPhraseSearcher(phrase_model=metadata['phrase_model'], config=metadata['configs'][key])
    cases = load_cases()
    if not request.config.getoption('--regression-full'):
        cases = cases[::QUICK_STRIDE]
    problems, extras = [], []
    for case in cases:
        if key in case['unstable_configs']:
            continue
        missing, changed, extra = compare(actual_matches(searcher, case['text']), case['expected'][key])
        if missing or changed:
            problems.append({'paragraph_id': case['paragraph_id'], 'missing': missing, 'changed': changed})
        if extra:
            extras.append({'paragraph_id': case['paragraph_id'], 'extra': extra})
    n_extra = sum(len(e['extra']) for e in extras)
    if extras:
        print(f'\n{key}: {n_extra} new matches in {len(extras)} paragraphs; first: {extras[0]}')
    assert not problems, (f"{sum(len(p['missing']) for p in problems)} matches missing and "
                          f"{sum(len(p['changed']) for p in problems)} with changed scores in "
                          f'{len(problems)} paragraphs; first: {problems[0]}')
    if request.config.getoption('--regression-exact'):
        assert not extras, f'{n_extra} new matches in {len(extras)} paragraphs; first: {extras[0]}'
