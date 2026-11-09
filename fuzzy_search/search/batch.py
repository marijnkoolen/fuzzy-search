"""Run a searcher's find_matches over many texts in parallel worker processes.

Workers are started with the 'spawn' method on every platform, so scripts that call
find_matches_batch with more than one worker need an ``if __name__ == '__main__':`` guard.
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Union

# The searcher of the current worker process, set once by _init_worker.
_worker_searcher = None


def _init_worker(searcher):
    global _worker_searcher
    _worker_searcher = searcher


def _find_matches_chunk(args):
    texts, kwargs = args
    return [_worker_searcher.find_matches(text, **kwargs) for text in texts]


def _chunks(items: List, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _relink_phrases(searcher, matches):
    """Point the phrase and variant of matches built in a worker back to the searcher's own objects."""
    model = searcher.phrase_model
    for match in matches:
        phrase = model.get_phrase(match.phrase.phrase_string)
        variant = model.get_phrase(match.variant.phrase_string)
        if phrase is not None:
            match.phrase = phrase
        if variant is not None:
            match.variant = variant
    return matches


def find_matches_batch(searcher, texts: Iterable[Union[str, Dict[str, str]]], workers: Union[None, int] = None,
                       chunk_size: int = 25, **kwargs) -> List[list]:
    """Return ``[searcher.find_matches(text, **kwargs) for text in texts]``, computed in parallel.

    :param searcher: the searcher whose find_matches is run; it is pickled once per worker
    :param texts: the texts (strings or dictionaries with a 'text' property) to search
    :param workers: the number of worker processes; None or 1 runs in this process,
        0 or a negative number uses one worker per CPU
    :param chunk_size: the number of texts sent to a worker per task
    :param kwargs: keyword arguments passed on to find_matches
    :return: one list of matches per text, in input order
    """
    texts = list(texts)
    if chunk_size < 1:
        raise ValueError(f'chunk_size must be at least 1, not {chunk_size}')
    if workers is not None and workers <= 0:
        workers = os.cpu_count() or 1
    if workers is None or workers == 1 or len(texts) <= chunk_size:
        return [searcher.find_matches(text, **kwargs) for text in texts]
    workers = min(workers, -(-len(texts) // chunk_size))
    context = multiprocessing.get_context('spawn')
    tasks = ((chunk, kwargs) for chunk in _chunks(texts, chunk_size))
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_init_worker, initargs=(searcher,)) as pool:
        for chunk_result in pool.map(_find_matches_chunk, tasks):
            results.extend(_relink_phrases(searcher, matches) for matches in chunk_result)
    return results
