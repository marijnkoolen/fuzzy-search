from typing import List, Union

from fuzzy_search.fuzzy_match import Match, MatchInContext
from fuzzy_search.fuzzy_phrase_searcher import FuzzyPhraseSearcher


class FuzzyContextSearcher(FuzzyPhraseSearcher):

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

    def add_match_context(self, match: Match, text: Union[str, dict], context_size: Union[None, int] = None,
                          use_prefix: bool = True, use_suffix: bool = True) -> MatchInContext:
        """Add context to a given match and its corresponding text document.

        :param match: a phrase match object
        :type match: Match
        :param text: the text that the match was taken from
        :type text: Union[str, dict]
        :param context_size: the size of the pre- and post context window
        :type context_size: int
        :param use_prefix: boolean to include prefix context or not
        :type use_prefix: bool
        :param use_suffix: boolean to include suffix context or not
        :type use_suffix: bool
        :return: the phrase match object with context
        :rtype: MatchInContext
        """
        if context_size is None:
            context_size = self.context_size
        prefix_size = context_size if use_prefix else 0
        suffix_size = context_size if use_suffix else 0
        return MatchInContext(match, text, prefix_size=prefix_size, suffix_size=suffix_size)

    def find_matches_in_context(self, match_in_context: MatchInContext,
                                use_word_boundaries: Union[None, bool] = None,
                                include_variants: Union[None, bool] = None,
                                filter_distractors: Union[None, bool] = None) -> List[Match]:
        """Use a MatchInContext object to find other phrases in the context of that match.

        :param match_in_context: a match phrase with context from the text that the match was taken from
        :type match_in_context: MatchInContext
        :param use_word_boundaries: boolean whether to adjust match strings to word boundaries
        :type use_word_boundaries: bool
        :param include_variants: boolean whether to include variants of phrases in matching
        :type include_variants: bool
        :param filter_distractors: boolean whether to remove matches that are closer to distractors than to their
        match phrases
        :type filter_distractors: bool
        :return: a list of match objects
        :rtype: List[Match]
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
