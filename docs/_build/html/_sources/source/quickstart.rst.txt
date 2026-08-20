Quick Start
===========

Installation
-------------

.. code-block:: bash

   pip install fuzzy-search

Basic usage
-----------

The most common workflow is: define a list of phrases (keywords or longer
phrases you want to find), build a :class:`~fuzzy_search.search.phrase_searcher.FuzzyPhraseSearcher`
from them, and search a text for fuzzy matches.

.. code-block:: python

   from fuzzy_search import make_searcher, default_config

   phrases = [
       "Provincien van Hollandt en Westvrieslandt",
       "Staten Generael",
   ]

   searcher = make_searcher(phrases, default_config)

   text = "de Staten Generaal der Vereenighde Nederlanden ende de Provintien van Hollandt en Westvrieslant"
   matches = searcher.find_matches(text)

   for match in matches:
       print(match.phrase.phrase_string, match.string, match.offset, match.levenshtein_similarity)

Each match is a :class:`~fuzzy_search.match.phrase_match.PhraseMatch` with the
matched phrase, the matched text, its offset, and similarity scores.

Tuning the fuzziness
---------------------

The ``config`` dict controls thresholds for character/ngram overlap and
Levenshtein similarity. Start from ``fuzzy_search.search.config.default_config``
and override individual keys, e.g.:

.. code-block:: python

   from fuzzy_search.search.config import default_config

   config = {**default_config, "char_match_threshold": 0.6, "ngram_size": 3}

Spelling variants and phrase models
------------------------------------

For more control, build a :class:`~fuzzy_search.phrase.phrase_model.PhraseModel`
directly, which lets you register known spelling variants per phrase:

.. code-block:: python

   from fuzzy_search.phrase.phrase_model import PhraseModel
   from fuzzy_search.search.phrase_searcher import FuzzyPhraseSearcher

   phrase_model = PhraseModel(phrases=[
       {"phrase": "Staten Generael", "variants": ["Staten Generaal", "Staeten Generael"]},
   ])
   searcher = FuzzyPhraseSearcher(phrase_model=phrase_model)

See the :doc:`fuzzy_search` API reference for the full set of searchers
(:class:`~fuzzy_search.search.token_searcher.FuzzyTokenSearcher`,
:class:`~fuzzy_search.search.context_searcher.FuzzyContextSearcher`,
:class:`~fuzzy_search.search.template_searcher.FuzzyTemplateSearcher`)
and the ``notebooks/`` directory in the repository for end-to-end examples
on historical and modern text.
