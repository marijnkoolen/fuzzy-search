# Proposal: Rust port, multi-core search and regression test set

Status: proposal, nothing ported yet. Measured on 2026-11-06 with fuzzy-search 3.0.4, Python 3.14 and a
10-core Mac. The determinism fix and the fixture were done on 2026-11-09 (see below).

## Summary

1. **Fix non-deterministic matching first. Done:** iteration in `get_skipmatch_candidates` is now sorted.
2. **Use Rust with PyO3 and maturin.** Port the skipgram and candidate pipeline, not just `text2skipgrams`.
3. **Add a process-based `find_matches_batch` now.** Replace its internals with Rust and rayon later.
4. **Build and commit the regression fixture after step 1,** and before any port work.

## Blocker: results depend on the searcher build

Two `FuzzyPhraseSearcher` objects with identical configuration can return different matches. A single
searcher is consistent with itself, but a rebuilt or unpickled one is not.

| Measurement | Result |
| --- | --- |
| Paragraphs differing over 4 builds, current code (`n2_s1_l0.5`, `n3_s2_l0.7`) | 11 and 3 of 400 |
| Differences seen | different variant credited; matches present or absent |
| Same check with ordered iteration (scratch patch, all 8 configs) | 0 of 400 |
| `PYTHONHASHSEED` 1 vs 2 | no change in output hash |

**Cause.** `Phrase` defines neither `__hash__` nor `__eq__`, so it hashes by memory address. The loop
`for phrase in skip_matches.matches` in `get_skipmatch_candidates` (`match/skip_match.py:264`) iterates a set
of these objects, and appends candidates of a phrase and its variants to one list. The later overlap filter
depends on that order. Exact matches and skipgram matches were identical across builds; candidates were not.

**Fix (applied).** `get_skipmatch_candidates` now iterates sorted by `phrase_string`. This changes some
results for existing users, so it needs a release note. On the fixture sample, match totals changed only for
`n2_s1_l0.5` (6,344 to 6,341) and `n2_s2_l0.5` (8,973 to 8,972); individual matches were not compared. Whether sorted order gives
the *best* tie-break is a separate question: it only makes the choice reproducible.

Rust could not reproduce address-dependent order, so without this fix an exact-match port cannot be verified.

## Language: Rust

- **PyO3 and maturin** keep the Python API intact and build wheels. All three tools are already installed here.
- **rayon** gives data parallelism without pickling, and PyO3 can release the GIL during the work.
- **C** offers no speed advantage worth the manual memory management and weaker parallel tooling.

## What to port

Profile of `find_matches` on 400 paragraphs (267k characters, 38 phrases, variants not enabled in this
profile). cProfile overhead is large (35s against 9.6s unprofiled), which inflates small Python functions, so
treat the shares as rough.

| Stage | `n2_s1_l0.5` | `n3_s2_l0.7` |
| --- | --- | --- |
| Candidate building (`get_skipmatch_phrase_candidates`, `candidate_match.py`) | 22.5s of 35.4s (63%) | 20.1s of 34.1s (59%) |
| Skipgram scan (`find_skipgram_matches`, includes `text2skipgrams` and `add_skip_match`) | 11.3s (32%) | 13.6s (40%) |

Candidate building is the largest cost: about 8 million skipgrams pass through `add_skip_match` and its
helpers one at a time. Porting only `text2skipgrams` would therefore give little, and crossing the Python/Rust
boundary per skipgram could even be slower.

**Port as one unit, called once per text:**

1. `text2skipgrams`
2. `find_skipgram_matches` and `SkipMatches`
3. `get_skipmatch_candidates`, `CandidatePartial` and its helpers, `filter_overlapping_phrase_candidates`

The Rust side should use integer phrase ids and byte-free character offsets, and return plain candidate
tuples. **Offsets must stay Unicode code points** to match Python slicing; the corpus contains non-ASCII text
such as `Præside` and `‛`.

**Keep in Python at first:** phrase model parsing, exact matching, `PhraseMatch`, thresholds and overlap
filtering. Move scoring later only if a new profile justifies it. `Levenshtein` is already compiled.

**Speed-up is not yet known.** Run a spike on steps 1 and 2 and re-measure before committing to the full port.

### Python behaviour a port must not copy blindly

- `first_best` is overwritten with a match in `filter_matches_by_overlap` (`match/match_offsets.py:39`), so
  `first_best=True` stops applying after the first overlap group.
- `max_offset` in `republic_phrase_model.py` is ignored: only `max_start_offset` is parsed. A phrase with
  `max_offset: 4` still matched at offset 100. Fix the key in that model, or support it here.

## Multi-core

Measured with one config on the same 400 paragraphs, with spawned worker processes and 25 paragraphs per task.
Times include worker start-up and building a searcher in each worker (0.5 to 1s), so they understate scaling on
long runs.

| Workers | Wall time | Speed-up |
| --- | --- | --- |
| 1 (no pool) | 9.8s | 1.0x |
| 2 | 5.85s | 1.7x |
| 4 | 3.85s | 2.5x |
| 8 | 3.49s | 2.8x |

Pickling is cheap: 0.02 to 0.09s and 1.2 to 5.3 MB per searcher. It is not the bottleneck I first suggested.

**Phase 1, Python:** add `find_matches_batch(texts, workers=None, chunk_size=..., **kwargs)`.

- Use `ProcessPoolExecutor` with an initializer that gives each worker one searcher.
- Return results in input order. Use `map`, not `as_completed`.
- Send tasks in chunks, and run in-process when `workers` is `None` or 1.
- Use the spawn start method on all platforms, and document that scripts need an `if __name__ == '__main__'` guard.
- Keep `find_matches` itself single-text. A single short text is too small to be worth splitting, and splitting
  a long one would lose matches across chunk borders.
- Requires the determinism fix, because each worker's searcher is a separate build.

**Phase 2, Rust:** run the ported pipeline over the batch with rayon, sharing one read-only index, with the GIL
released. The Python post-processing stays serial until it is ported too, which will limit scaling.
The `find_matches_batch` signature does not change.

## Regression test set

- `test/regression/build_phrase_regression_set.py` builds the fixture.
- `test/regression/test_phrase_regression.py` compares results with it, and skips if no fixture exists.

**Data.** `paragraphs/` holds no text, so paragraphs are rebuilt from `lines/`:

- Marginal notes are left out.
- Lines are joined with a space, and a trailing `-` or `=` is joined without one.
- A line split over fragments stays whole in its primary paragraph (1.25% of paragraphs in a 400-paragraph sample).

The fixture stores the rebuilt text, so tests need neither the dataset nor the sibling repository. It also
stores a copy of the phrase model.

**Sampling.** Per inventory, 30 resolution paragraphs and 10 others are drawn with a fixed seed. Openings and
attendance lists are included because phrases such as `PRAESIDE,` occur there. A paragraph is kept when it adds an
example for a (config, phrase) pair under the cap of 25, plus 200 paragraphs without matches. Every kept paragraph
stores expected results for all 8 configs (`ngram_size` 2 and 3, `skip_size` 1 and 2, `levenshtein_threshold` 0.5
and 0.7, variants on). The phrases are `session_phrase_model` plus `resolution_phrase_model`.

**Result after the fix (2026-11-09).** The fixture is in `test/regression/fixtures/`.

| Measure | Value |
| --- | --- |
| Kept paragraphs | 1,259, including 200 without matches |
| Phrases with examples | 38 of 38 in every config |
| Paragraphs flagged unstable over 4 extra builds | 0 |
| Fixture size | 971 KB compressed, plus 8 KB metadata |
| Test result | 8 of 8 configs pass, on 2 runs (about 8 minutes each) |

**Result of the full run before the fix, kept for comparison:**

| Measure | Value |
| --- | --- |
| Sampled and kept paragraphs | 17,720 sampled, 1,259 kept |
| Phrases with examples | 38 of 38 in every config |
| Matches per config | 3,223 to 8,973 |
| Paragraphs flagged unstable | 66 |
| Fixture size | 971 KB compressed, plus 8 KB metadata |
| Build time | 21 minutes with 6 workers |
| Test time | about 8 minutes |
| Test result, 3 runs | **7 of 8 configs fail each time** |

Those failures came from instability that the 4 extra builds did not catch.

To build, from the repository root:

```bash
PYTHONPATH=. uv run --no-project --python .venv/bin/python --with pyarrow python test/regression/build_phrase_regression_set.py
```

To test:

```bash
.venv/bin/python -m pytest test/regression
```

`unstable_configs` is now empty everywhere, so the stability check could become a hard build failure. Consider a smaller default fixture or `pytest -m` marker, since 8 minutes is slow for routine runs.

## Order of work

1. Determinism fix and fixture: done. Still to do: the `first_best` and `max_offset` fixes. Changing them
   changes results, so rebuild the fixture afterwards.
2. Release as a behaviour change.
3. Add `find_matches_batch` in Python, with a test that it equals serial results.
4. Rust spike on the skipgram scan, and re-profile.
5. Port the candidate pipeline, using the fixture as the acceptance test.
6. Move batch parallelism to rayon, and ship wheels through maturin in CI.

## Open questions

- Should ties between variants have a meaningful rule (for example prefer the main phrase), rather than just sorted order?
- Should matches be restricted by each phrase's `start_year` and `end_year`? The fixture ignores these fields, as
  `FuzzyPhraseSearcher` does not filter on them in this set-up.
- Does a pure Python fallback stay supported once Rust exists? Keeping it doubles the logic to maintain, but makes
  installation easier where no wheel is available.
