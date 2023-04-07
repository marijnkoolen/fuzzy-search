__version__ = '1.6.0'

from fuzzy_search.fuzzy_phrase_searcher import FuzzyPhraseSearcher
from fuzzy_search.fuzzy_phrase_model import PhraseModel
from fuzzy_search.fuzzy_phrase_searcher import default_config


def make_searcher(phrases: any, config):
    phrase_model = PhraseModel(phrases, config)
    searcher = FuzzyPhraseSearcher(phrase_model=phrase_model, config=config)
    return searcher

