import re
import fuzzy_search.fuzzy_patterns as fuzzy_patterns
from fuzzy_search.fuzzy_keyword_searcher import FuzzyKeywordSearcher


#######################################
# Functions for person name searching #
#######################################


def make_search_context_patterns(context_string, person_name_patterns, context_patterns):
    return fuzzy_patterns.make_search_context_patterns(context_string, person_name_patterns, context_patterns)


def find_patterns_in_context(context, patterns, context_patterns=None):
    for search_context in make_search_context_patterns(context["match_string"], patterns, context_patterns):
        # print("search_context:", search_context)
        for pattern_re_match in re.finditer(search_context["pattern"], context["match_term_in_context"]):
            yield pattern_re_match, search_context


def get_term_context(text, term_match, context_size=20, before_context=True, after_context=True):
    context = make_base_context(term_match)
    adjust_context_offset(context, context_size, before_context, after_context)
    add_context_text(context, text)
    return context


class FuzzyContextSearcher(FuzzyKeywordSearcher):

    def __init__(self, config):
        super().__init__(config)
        self.pattern_names = fuzzy_patterns.list_pattern_names(pattern_type=None)
        self.context_patterns = fuzzy_patterns.get_context_patterns(None)
        self.context_size = 100
        self.configure_context(config)

    def configure_context(self, config):
        if "pattern_type" in config:
            self.pattern_names = fuzzy_patterns.list_pattern_names(config["pattern_type"])
        if "context_type" in config:
            self.pattern_names = fuzzy_patterns.get_context_patterns(config["context_type"])
        if "context_size" in config:
            self.context_size = config["context_size"]

    def set_pattern_names_by_type(self, pattern_type):
        self.pattern_names = fuzzy_patterns.list_pattern_names(pattern_type)

    def set_pattern_names(self, pattern_names):
        for pattern_name in pattern_names:
            if pattern_name not in fuzzy_patterns.pattern_definitions.keys():
                raise KeyError("Pattern name does not exist")
        self.pattern_names = pattern_names

    def set_context_pattern_types(self, context_pattern_types):
        for context_pattern_type in context_pattern_types:
            if context_pattern_type not in fuzzy_patterns.context_pattern.keys():
                raise KeyError("Context pattern type does not exist")
        self.context_pattern_types = context_pattern_types

    def find_candidates_in_context(self, context_match, keyword=None, ngram_size=2,
                                   use_word_boundaries=None, match_initial_char=False,
                                   include_variants=False, filter_distractors=False):
        text = context_match["match_term_in_context"]
        self.find_candidates(text, keyword=keyword, ngram_size=ngram_size,
                             use_word_boundaries=use_word_boundaries,
                             match_initial_char=match_initial_char,
                             include_variants=include_variants,
                             filter_distractors=filter_distractors)
        self.update_candidate_offsets(context_match)
        return self.candidates["accept"]

    def update_candidate_offsets(self, context_match):
        for candidate in self.candidates["accept"]:
            candidate["match_offset"] += context_match["start_offset"]


def add_context_text(context, text):
    if context["end_offset"] > len(text):
        context["end_offset"] = len(text)
    context_text = text[context["start_offset"]:context["end_offset"]]
    context["match_term_in_context"] = context_text


def make_base_context(term_match):
    return {
        "match_keyword": term_match["match_keyword"],
        "match_term": term_match["match_term"],
        "match_string": term_match["match_string"],
        "match_offset": term_match["match_offset"],
        "start_offset": term_match["match_offset"],
        "end_offset": term_match["match_offset"] + len(term_match["match_string"]),
    }


def adjust_context_offset(context, context_size, before_context, after_context):
    if before_context:
        context["start_offset"] = context["match_offset"] - context_size
    if after_context:
        context["end_offset"] = context["match_offset"] + len(context["match_string"]) + context_size + 1
    if context["match_offset"] < context_size:
        context["start_offset"] = 0


if __name__ == "__main__":
    sample_text = "A'nthony van der Truyn en Adriaen Bosman, Makelaers tot Rotterdam, prefenteren, uyt de Hint te verkopen etfn curieufc Party opreckw ?al somfl'e Schalyen of Leyen."
