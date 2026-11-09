"""Build the FuzzyPhraseSearcher regression fixture from REPUBLIC paragraphs.

Needs pyarrow, which is not a dependency of fuzzy-search itself. Usage is in docs/rust_port_proposal.md.
Every selected paragraph stores the expected matches under every configuration.
"""
import argparse
import gzip
import importlib.util
import itertools
import json
import multiprocessing
import platform
import random
from collections import Counter, defaultdict
from pathlib import Path

import Levenshtein
import pyarrow.parquet as pq

from fuzzy_search._version import __version__
from fuzzy_search.search.phrase_searcher import FuzzyPhraseSearcher

DEFAULT_DATA_DIR = Path('/Users/Shared/claude_projects/data_scopes/shared-data/NL-HaNA_1.01.02/'
                        'republic_segmentation_v5')
DEFAULT_MODEL_FILE = Path(__file__).resolve().parents[3] / 'republic-project/republic/model/republic_phrase_model.py'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'fixtures'
RESOLUTION_TYPE = 'resolution_paragraph'
NON_TEXT_UNIT_TYPE = 'marginal_note'
LINE_COLUMNS = ['text', 'seq', 'unit_type', 'paragraph_id', 'n_fragments']

_worker_searchers = {}


def make_configs():
    return [{'ngram_size': ngram_size, 'skip_size': skip_size, 'levenshtein_threshold': threshold,
             'include_variants': True}
            for ngram_size, skip_size, threshold in itertools.product((2, 3), (1, 2), (0.5, 0.7))]


def config_key(config):
    return f"n{config['ngram_size']}_s{config['skip_size']}_l{config['levenshtein_threshold']}"


def load_phrase_model(model_file):
    spec = importlib.util.spec_from_file_location('republic_phrase_model', model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    phrases = {}
    # The same phrase may occur in several model lists; the first definition wins.
    for entry in module.session_phrase_model + module.resolution_phrase_model:
        phrases.setdefault(entry['phrase'], entry)
    return list(phrases.values())


def join_lines(lines):
    text = ''
    for line in lines:
        line = line.strip()
        if text.endswith(('-', '=')):
            text = text[:-1] + line
        elif text:
            text += ' ' + line
        else:
            text = line
    return text


def read_inventory_paragraphs(inv_dir):
    """Rebuild paragraph texts of one inventory from its lines.

    A line split over several fragments is kept whole in its primary paragraph, because
    fragment offsets are not line-local (see the dataset README). Fixtures store the text,
    so this choice only affects which inputs are tested, not how results are compared.
    """
    inventory = inv_dir.name.split('=')[1]
    rows = pq.read_table(inv_dir / 'part-0.parquet', columns=LINE_COLUMNS).to_pylist()
    grouped = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['seq']):
        if row['paragraph_id'] and row['text'] and row['unit_type'] != NON_TEXT_UNIT_TYPE:
            grouped[row['paragraph_id']].append(row)
    return [{'paragraph_id': paragraph_id, 'inventory': inventory, 'type': rows_[0]['unit_type'],
             'has_split_line': any(row['n_fragments'] > 1 for row in rows_),
             'text': join_lines([row['text'] for row in rows_])}
            for paragraph_id, rows_ in sorted(grouped.items())]


def sample_paragraphs(data_dir, per_inventory, other_per_inventory, seed, max_inventories=None):
    rng = random.Random(seed)
    inv_dirs = sorted((data_dir / 'lines').glob('inv=*'), key=lambda p: int(p.name.split('=')[1]))
    if max_inventories:
        inv_dirs = sorted(rng.sample(inv_dirs, min(max_inventories, len(inv_dirs))),
                          key=lambda p: int(p.name.split('=')[1]))
    sample = []
    for inv_dir in inv_dirs:
        paragraphs = [p for p in read_inventory_paragraphs(inv_dir) if p['text']]
        resolutions = [p for p in paragraphs if p['type'] == RESOLUTION_TYPE]
        others = [p for p in paragraphs if p['type'] != RESOLUTION_TYPE]
        sample += rng.sample(resolutions, min(per_inventory, len(resolutions)))
        sample += rng.sample(others, min(other_per_inventory, len(others)))
    return sample


def serialise_match(match):
    return {'phrase': match.phrase.phrase_string, 'variant': match.variant.phrase_string,
            'string': match.string, 'offset': match.offset, 'end': match.end, 'label': match.label,
            'char_match': match.character_overlap, 'ngram_match': match.ngram_overlap,
            'levenshtein_similarity': match.levenshtein_similarity}


def match_sort_key(match):
    return match['offset'], match['end'], match['phrase'], match['variant'], match['string']


def run_config_on_text(searcher, text):
    matches = [serialise_match(match) for match in searcher.find_matches(text)]
    return sorted(matches, key=match_sort_key)


def _run_chunk(task):
    phrase_model, config, chunk, build = task
    key = config_key(config)
    # Older searchers stay cached, so each build gets its own memory addresses.
    if (key, build) not in _worker_searchers:
        _worker_searchers[(key, build)] = FuzzyPhraseSearcher(phrase_model=phrase_model, config=config)
    searcher = _worker_searchers[(key, build)]
    return key, [(index, run_config_on_text(searcher, text)) for index, text in chunk]


def search_all(paragraphs, phrase_model, configs, workers, chunk_size=50, build=0):
    results = [{} for _ in paragraphs]
    indexed = [(index, paragraph['text']) for index, paragraph in enumerate(paragraphs)]
    chunks = [indexed[i:i + chunk_size] for i in range(0, len(indexed), chunk_size)]
    tasks = [(phrase_model, config, chunk, build) for config in configs for chunk in chunks]
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        for done, (key, chunk_results) in enumerate(pool.imap_unordered(_run_chunk, tasks), start=1):
            for index, matches in chunk_results:
                results[index][key] = matches
            if done % 100 == 0 or done == len(tasks):
                print(f'{done}/{len(tasks)} chunks done', flush=True)
    return results


def mark_unstable_configs(cases, phrase_model, configs, workers, num_builds):
    """Flag results that change between searcher builds, which the regression test must skip.

    Results of a config can depend on the searcher build (see docs/rust_port_proposal.md),
    so a fixture without this check would make the test flaky.
    """
    for case in cases:
        case['unstable_configs'] = []
    for build in range(1, num_builds + 1):
        for case, rerun in zip(cases, search_all(cases, phrase_model, configs, workers, build=build)):
            for key, matches in rerun.items():
                if matches != case['expected'][key] and key not in case['unstable_configs']:
                    case['unstable_configs'].append(key)
    for case in cases:
        case['unstable_configs'].sort()


def select_cases(paragraphs, results, max_per_phrase, num_no_match, seed):
    """Keep a paragraph if it adds an example of a (config, phrase) pair still under the cap."""
    order = list(range(len(paragraphs)))
    random.Random(seed).shuffle(order)
    counts = Counter()
    selected, no_match = [], []
    for index in order:
        pairs = {(key, match['phrase']) for key, matches in results[index].items() for match in matches}
        if not pairs:
            if len(no_match) < num_no_match:
                no_match.append(index)
            continue
        if any(counts[pair] < max_per_phrase for pair in pairs):
            selected.append(index)
            counts.update(pairs)
    return sorted(selected + no_match)


def write_fixture(output_dir, name, cases, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f'{name}.jsonl.gz', 'wb') as raw:
        # mtime=0 keeps rebuilt fixtures byte-identical, so git diffs are meaningful.
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as fh:
            for case in cases:
                fh.write((json.dumps(case, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8'))
    with open(output_dir / f'{name}.meta.json', 'wt', encoding='utf-8') as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=1, sort_keys=True)


def summarise(cases, phrase_model, configs):
    per_config_phrase = defaultdict(Counter)
    for case in cases:
        for key, matches in case['expected'].items():
            per_config_phrase[key].update(match['phrase'] for match in matches)
    all_phrases = {entry['phrase'] for entry in phrase_model}
    found = set().union(*(set(counter) for counter in per_config_phrase.values())) if per_config_phrase else set()
    return {
        'num_cases': len(cases),
        'num_no_match_cases': sum(not any(case['expected'].values()) for case in cases),
        'num_cases_with_unstable_config': sum(bool(case['unstable_configs']) for case in cases),
        'unstable_case_config_pairs': Counter(key for case in cases for key in case['unstable_configs']),
        'num_matches_per_config': {config_key(c): sum(per_config_phrase[config_key(c)].values()) for c in configs},
        'phrases_with_examples_per_config': {config_key(c): len(per_config_phrase[config_key(c)]) for c in configs},
        'phrases_without_examples': sorted(all_phrases - found),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--model-file', type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--name', default='phrase_regression')
    parser.add_argument('--per-inventory', type=int, default=30, help='resolution paragraphs sampled per inventory')
    parser.add_argument('--other-per-inventory', type=int, default=10,
                        help='opening, attendance and other paragraphs sampled per inventory')
    parser.add_argument('--max-inventories', type=int, default=None, help='sample inventories, for quick runs')
    parser.add_argument('--max-per-phrase', type=int, default=25, help='example cap per (config, phrase) pair')
    parser.add_argument('--num-no-match', type=int, default=200, help='paragraphs without any match to keep')
    parser.add_argument('--stability-builds', type=int, default=4, help='extra searcher builds to compare results with')
    parser.add_argument('--allow-unstable', action='store_true',
                        help='write the fixture even if results differ between searcher builds')
    parser.add_argument('--workers', type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument('--seed', type=int, default=20261106)
    return parser.parse_args()


def main():
    args = parse_args()
    phrase_model = load_phrase_model(args.model_file)
    configs = make_configs()
    paragraphs = sample_paragraphs(args.data_dir, args.per_inventory, args.other_per_inventory, args.seed,
                                   args.max_inventories)
    print(f'{len(phrase_model)} phrases, {len(configs)} configs, {len(paragraphs)} sampled paragraphs', flush=True)
    results = search_all(paragraphs, phrase_model, configs, args.workers)
    cases = [{**paragraphs[index], 'expected': results[index]}
             for index in select_cases(paragraphs, results, args.max_per_phrase, args.num_no_match, args.seed)]
    print(f'{len(cases)} cases selected, checking stability over {args.stability_builds} extra builds', flush=True)
    mark_unstable_configs(cases, phrase_model, configs, args.workers, args.stability_builds)
    metadata = {
        'fuzzy_search_version': __version__, 'levenshtein_version': Levenshtein.__version__,
        'python_version': platform.python_version(), 'seed': args.seed,
        'data_dir': str(args.data_dir), 'num_sampled_paragraphs': len(paragraphs),
        'sampling': {'per_inventory': args.per_inventory, 'other_per_inventory': args.other_per_inventory,
                     'max_inventories': args.max_inventories, 'max_per_phrase': args.max_per_phrase,
                     'num_no_match': args.num_no_match},
        'configs': {config_key(config): config for config in configs},
        'phrase_model': phrase_model,
        'summary': summarise(cases, phrase_model, configs),
    }
    print(json.dumps(metadata['summary'], ensure_ascii=False, indent=1))
    num_unstable = metadata['summary']['num_cases_with_unstable_config']
    if num_unstable and not args.allow_unstable:
        raise SystemExit(f'{num_unstable} cases differ between searcher builds; fixture not written '
                         f'(use --allow-unstable to write it anyway)')
    write_fixture(args.output_dir, args.name, cases, metadata)


if __name__ == '__main__':
    main()
