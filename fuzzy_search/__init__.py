"""fuzzy_search: a library for fuzzy phrase search in noisy and historical text.

Re-exports the most commonly used public classes (``FuzzyPhraseSearcher``,
``FuzzyTokenSearcher``, ``PhraseMatch``, ``PhraseModel``, ``default_config``) and
provides the ``make_searcher`` convenience function for building a phrase searcher
from a list of phrases and a config.
"""

from fuzzy_search._version import __version__
from fuzzy_search.search.config import default_config
from fuzzy_search.search.phrase_searcher import FuzzyPhraseSearcher
from fuzzy_search.search.token_searcher import FuzzyTokenSearcher
from fuzzy_search.match.phrase_match import PhraseMatch
from fuzzy_search.phrase.phrase_model import PhraseModel


def make_searcher(phrases: any, config):
    """Builds a FuzzyPhraseSearcher for a given list of phrases and configuration.

    Args:
        phrases (any): The phrases to search for, as accepted by ``PhraseModel``.
        config: A configuration object/dict for the phrase model and searcher.

    Returns:
        FuzzyPhraseSearcher: A searcher configured with the built phrase model.
    """
    phrase_model = PhraseModel(phrases, config)
    searcher = FuzzyPhraseSearcher(phrase_model=phrase_model, config=config)
    return searcher
