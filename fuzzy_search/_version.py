"""Single source of truth for the package version.

Kept in its own module (rather than in ``fuzzy_search/__init__.py``) so that
submodules such as :mod:`fuzzy_search.search.searcher` can read the version
without importing the top-level ``fuzzy_search`` package, which would create
a circular import (``fuzzy_search/__init__.py`` itself imports those
submodules).
"""

__version__ = "3.0.0"
