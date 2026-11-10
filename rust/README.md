# fuzzy-search-rs

Optional Rust backend for `FuzzyPhraseSearcher` in [fuzzy-search](../README.md). It returns exactly the
same matches as the Python implementation, about 35 times faster on one thread, and runs batches on
several threads without the GIL.

## Wheels

The extension is built as an abi3 wheel: one wheel per platform works on every CPython from 3.10 on.
The workflow `.github/workflows/rust-wheels.yml` builds and tests wheels for Linux (x86_64 and aarch64,
manylinux2014; x86_64 musllinux), macOS (universal2: Intel and Apple silicon) and Windows (x64 and
arm64), and uploads them as workflow artifacts. Publishing to PyPI is not automated.

## Build and install

Needs a Rust toolchain and [maturin](https://www.maturin.rs/). From this directory:

```bash
maturin build --release
```

Other platforms from macOS (with rustup's toolchain first on `PATH`; Homebrew's `rust` has only the
host target):

```bash
rustup target add x86_64-apple-darwin x86_64-unknown-linux-gnu
```

```bash
maturin build --release --target universal2-apple-darwin
```

```bash
pip install ziglang && maturin build --release --target x86_64-unknown-linux-gnu --zig --compatibility manylinux2014
```

Cross-building aarch64 Linux with zig currently fails at link time (zig rejects rustc's
`--fix-cortex-a53-843419` flag), and Windows wheels need a Windows machine; CI builds both natively.

```bash
uv pip install --python ../.venv/bin/python --reinstall target/wheels/fuzzy_search_rs-*.whl
```

or, with pip, `pip install target/wheels/fuzzy_search_rs-*.whl`.

## Use

Select the backend in the searcher config:

```python
from fuzzy_search import FuzzyPhraseSearcher

searcher = FuzzyPhraseSearcher(phrase_model=phrases, config={'backend': 'auto'})
matches = searcher.find_matches(text)                          # one text
results = searcher.find_matches_batch(texts, workers=0)        # many texts, one thread per CPU
```

- `'auto'` (default): Rust when installed and possible, otherwise Python.
- `'python'`: the Python implementation.
- `'rust'`: the Rust backend; raises `RuntimeError` if it is not installed or cannot reproduce the
  searcher's settings.

With the Rust backend, `find_matches_batch(texts, workers=n)` uses `n` threads (0: one per CPU) instead
of worker processes, so no `if __name__ == '__main__':` guard is needed.

## What runs in Rust

`find_matches` runs exact matching in Python, then Rust does, per text:

1. the skipgram scan (`Searcher.find_skipgram_matches`),
2. candidate building (`get_skipmatch_candidates`, `CandidatePartial`),
3. word-boundary adjustment (`adjust_match_offsets`) and per-phrase overlap filtering
   (`filter_overlapping_phrase_candidates`),
4. scoring (`PhraseMatch.add_scores`: character overlap, ngram overlap, `Levenshtein.ratio`).

Python then builds `PhraseMatch` objects only for candidates that pass the thresholds, and runs the
remaining steps (distractors, combining with exact matches, offset thresholds, overlap filtering).

The Rust code mirrors the Python code step by step, including quirks such as group ordering, the
`next_start_offset and ...` truthiness test and `Counter` entries that drop to zero. Offsets are Unicode
code points, as in Python slicing.

## Limits and differences

- `ngram_size` up to 6 (skipgrams are packed into 128-bit keys). Larger sizes fall back to Python
  (`'auto'`) or raise (`'rust'`).
- Phrases with `max_end_offset` fall back to Python, which raises an `AttributeError` for them.
- `debug > 0` always runs the Python code, so its debug output is unchanged.
- With `ignorecase`, texts whose lowercase has a different length (such as `İ`) or that contain a capital
  sigma `Σ` are scored in Python, because there slicing the lowercased text differs from lowercasing a slice.
- Where Python would raise `ZeroDivisionError` or `IndexError` on degenerate input (a phrase shorter
  than `ngram_size`), the Rust backend raises `ValueError`.

## Verification

- `test/test_search_rust_backend.py` compares both backends over a grid of settings and texts, including
  threaded batches, and checks backend selection and pickling.
- The regression fixture runs with either backend:

```bash
.venv/bin/python -m pytest test/regression --regression-backend rust --regression-exact --regression-full
```
