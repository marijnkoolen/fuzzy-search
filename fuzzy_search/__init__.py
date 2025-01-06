__version__ = '2.4.1'

from fuzzy_search.search.config import default_config
from fuzzy_search.search.phrase_searcher import FuzzyPhraseSearcher
from fuzzy_search.search.token_searcher import FuzzyTokenSearcher
from fuzzy_search.match.phrase_match import PhraseMatch
from fuzzy_search.phrase.phrase_model import PhraseModel


def make_searcher(phrases: any, config):
    phrase_model = PhraseModel(phrases, config)
    searcher = FuzzyPhraseSearcher(phrase_model=phrase_model, config=config)
    return searcher
