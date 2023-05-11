from typing import List, Union

from fuzzy_search.match.phrase_match import PhraseMatch, PhraseMatchInContext
from fuzzy_search.search.phrase_searcher import FuzzyPhraseSearcher


class FuzzyContextSearcher(FuzzyPhraseSearcher):
    """

    Attributes
    ----------
    context_size : int
    """

    def __init__(self, config: Union[dict, None] = None):
        super().__init__(config)
        self.context_size = 100
        if config is not None:
            self.configure_context(config)

    def configure_context(self, config: dict) -> None:
        """Configure the context searcher.

        :param config: a dictionary with configuration parameters to override the defaults
        :type config: dict
        """
        super().configure(config)
        if "context_size" in config:
            self.context_size = config["context_size"]

    def add_match_context(self, match: PhraseMatch, text: Union[str, dict], context_size: Union[None, int] = None,
                          prefix_size: Union[None, int] = None,
                          suffix_size: Union[None, int] = None) -> PhraseMatchInContext:
        """Add context to a given match and its corresponding text document.

        :param match: a phrase match object
        :type match: PhraseMatch
        :param text: the text that the match was taken from
        :type text: Union[str, dict]
        :param context_size: the size of the pre- and suffix window
        :type context_size: int
        :param prefix_size: size of the prefix context
        :type prefix_size: Union[None, int]
        :param suffix_size: size of the suffix context
        :type suffix_size: Union[None, int]
        :return: the phrase match object with context
        :rtype: PhraseMatchInContext
        """
        if context_size is None:
            context_size = self.context_size
        prefix_size = prefix_size if prefix_size is not None else context_size
        suffix_size = suffix_size if suffix_size is not None else context_size
        return PhraseMatchInContext(match, text, prefix_size=prefix_size, suffix_size=suffix_size)

    def find_matches(self, text: Union[str, dict],
                     use_word_boundaries: Union[None, bool] = None,
                     allow_overlapping_matches: bool = True,
                     include_variants: bool = None,
                     filter_distractors: bool = None,
                     prefix_size: Union[None, int] = None,
                     suffix_size: Union[None, int] = None,
                     skip_exact_matching: bool = None) -> List[PhraseMatchInContext]:
        """Find fuzzy matches for registered phrases and add context around match string. This extends
        the find_matches function of the FuzzyPhraseSearcher by adding local context to each match.

        :param text: the text (string or dictionary with 'text' property) to find fuzzy matching phrases in.
        :type text: Union[str, Dict[str, str]]
        :param use_word_boundaries: use word boundaries in determining match boundaries
        :type use_word_boundaries: bool
        :param allow_overlapping_matches: boolean flag for whether to allow matches to overlap in their text ranges
        :type allow_overlapping_matches: bool
        :param include_variants: boolean flag for whether to include phrase variants for finding matches
        :type include_variants: bool
        :param filter_distractors: boolean flag for whether to remove phrase matches that better match distractors
        :type filter_distractors: bool
        :param prefix_size: the size of the prefix context window
        :type prefix_size: Union[None, int]
        :param suffix_size: the size of the suffix context window
        :type suffix_size: Union[None, int]
        :param skip_exact_matching: boolean flag whether to skip the exact matching step
        :type skip_exact_matching: Union[None, bool]
        :return: a list of phrases matches with text surrounding the match string
        :rtype: PhraseMatchInContext
        """
        matches = super().find_matches(text, use_word_boundaries=use_word_boundaries,
                                       allow_overlapping_matches=allow_overlapping_matches,
                                       include_variants=include_variants, filter_distractors=filter_distractors,
                                       skip_exact_matching=skip_exact_matching)
        return [self.add_match_context(match, text, prefix_size=prefix_size,
                                       suffix_size=suffix_size) for match in matches]

    def find_matches_in_context(self, match_in_context: PhraseMatchInContext,
                                use_word_boundaries: Union[None, bool] = None,
                                include_variants: Union[None, bool] = None,
                                filter_distractors: Union[None, bool] = None) -> List[PhraseMatch]:
        """Use a MatchInContext object to find other phrases in the context of that match.

        :param match_in_context: a match phrase with context from the text that the match was taken from
        :type match_in_context: PhraseMatchInContext
        :param use_word_boundaries: boolean whether to adjust match strings to word boundaries
        :type use_word_boundaries: bool
        :param include_variants: boolean whether to include variants of phrases in matching
        :type include_variants: bool
        :param filter_distractors: boolean whether to remove matches that are closer to distractors
        :type filter_distractors: bool
        :return: a list of match objects
        :rtype: List[PhraseMatch]
        """
        # override searcher configuration if a boolean is passed
        use_word_boundaries = self.use_word_boundaries if use_word_boundaries is None else use_word_boundaries
        include_variants = self.include_variants if include_variants is None else include_variants
        filter_distractors = self.filter_distractors if filter_distractors is None else filter_distractors
        # search the context of the original match
        context_matches = self.find_matches(match_in_context.context, use_word_boundaries=use_word_boundaries,
                                            include_variants=include_variants,
                                            filter_distractors=filter_distractors)
        # recalculate the match offset with respect to the original text
        for match in context_matches:
            match.offset += match_in_context.context_start
            match.end += match_in_context.context_start
        return context_matches


if __name__ == "__main__":
    sample_text = "A'nthony van der Truyn en Adriaen Bosman, Makelaers tot Rotterdam, prefenteren," + \
                  "uyt de Hint te verkopen etfn curieufc Party opreckw ?al somfl'e Schalyen of Leyen."
