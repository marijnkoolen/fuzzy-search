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


@pytest.mark.parametrize('key', sorted(load_metadata()['configs']) if META_FILE.exists() else [])
def test_matches_equal_fixture(key):
    metadata = load_metadata()
    searcher = FuzzyPhraseSearcher(phrase_model=metadata['phrase_model'], config=metadata['configs'][key])
    differences = []
    for case in load_cases():
        if key in case['unstable_configs']:
            continue
        actual = actual_matches(searcher, case['text'])
        expected = case['expected'][key]
        same = [comparable(m) for m in actual] == [comparable(m) for m in expected] and all(
            a[field] == pytest.approx(e[field], abs=1e-9)
            for a, e in zip(actual, expected) for field in SCORE_FIELDS)
        if not same:
            differences.append({'paragraph_id': case['paragraph_id'], 'expected': expected, 'actual': actual})
    assert not differences, f'{len(differences)} paragraphs differ; first: {differences[0]}'
