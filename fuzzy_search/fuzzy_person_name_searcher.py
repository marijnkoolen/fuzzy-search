import fuzzy_search.fuzzy_patterns as fuzzy_patterns
import re
from fuzzy_search.fuzzy_context_searcher import FuzzyContextSearcher, find_patterns_in_context
from fuzzy_search.fuzzy_keyword_searcher import score_levenshtein_distance


#######################################
# Functions for person name searching #
#######################################


def find_person_names_in_context(context, person_name_patterns=None, context_patterns=None):
    name_matches = []
    if not person_name_patterns:
        person_name_patterns = fuzzy_patterns.get_search_patterns(pattern_type="dutch_person_name")
    if not context_patterns:
        context_patterns = fuzzy_patterns.get_context_patterns("person_name")
    for name_re_match, search_context in find_patterns_in_context(context, person_name_patterns, context_patterns):
        parse_person_name_in_context_matches(name_re_match, search_context, context, name_matches)
    return name_matches


def find_person_names_in_text(text, person_name_patterns=None):
    # NOT FINISHED, NOT WORKING YET!
    name_matches = []
    if not person_name_patterns:
        person_name_patterns = fuzzy_patterns.get_search_patterns(pattern_type="dutch_person_name")
    for pattern_name in person_name_patterns:
        search_pattern = person_name_patterns[pattern_name]
        for name_re_match in re.finditer(search_pattern["pattern"], text):
            parse_person_name_in_text_matches(name_re_match, search_pattern, text, name_matches)
    return name_matches


def remove_close_distance_names(close_distance_names, name1, name2):
    # print("Removing close_distance_names:", name1, "\t", name2)
    close_distance_names[name1].remove(name2)
    close_distance_names[name2].remove(name1)


class FuzzyPersonNameSearcher(FuzzyContextSearcher):

    def find_close_distance_names(self, name_list, max_distance_ratio=0.3):
        # TODO:
        # - make a specific version for person name:
        # - consider use of initials for first or last name, e.g. A. Doelman
        # - consider use of only last name, e.g. Doelman
        close_distance_names = self.find_close_distance_keywords(name_list, max_distance_ratio=max_distance_ratio)
        max_distance_ratio = 0.5
        self.preferred = {}
        self.filter_distant_first_names(close_distance_names, max_distance_ratio)
        self.filter_distant_last_names(close_distance_names, max_distance_ratio)
        return close_distance_names

    def filter_distant_first_names(self, close_distance_names, max_distance_ratio):
        for name1 in close_distance_names:
            first_name1 = name1.split(" ")[0]
            for name2 in close_distance_names[name1]:
                first_name2 = name2.split(" ")[0]
                # print("comparing first names:", first_name1, first_name2)
                if self.filter_distant_names(first_name1, first_name2, max_distance_ratio):
                    remove_close_distance_names(close_distance_names, name1, name2)

    def filter_distant_last_names(self, close_distance_names, max_distance_ratio):
        for name1 in close_distance_names:
            last_name1 = name1.split(" ")[-1]
            for name2 in close_distance_names[name1]:
                last_name2 = name2.split(" ")[-1]
                # print("comparing last names:", last_name1, last_name2)
                if self.filter_distant_names(last_name1, last_name2, max_distance_ratio):
                    remove_close_distance_names(close_distance_names, name1, name2)

    def filter_distant_names(self, name1, name2, max_distance_ratio):
        distance = score_levenshtein_distance(name1, name2)
        # print("distance:", distance, "ratio 1:", distance / len(name1), "ratio 2:", distance / len(name2))
        if distance / len(name1) > max_distance_ratio:
            return True
        elif distance / len(name2) > max_distance_ratio:
            return True
        else:
            return False


def is_new_name_match(name_match, known_matches):
    for known_match in known_matches:
        if known_match["name_string"] == name_match["name_string"]:
            if known_match["match_offset"] == name_match["match_offset"]:
                return False
    return True


def add_name_match_if_new(name_match, known_matches):
    if is_new_name_match(name_match, known_matches):
        known_matches.append(name_match)


def parse_person_name_in_context_matches(name_re_match, search_context, context, name_matches):
    match_offset = context["start_offset"] + name_re_match.start()
    whole_match = name_re_match.group(0)
    for pattern_group_index in search_context["group_indices"]:
        if not name_re_match.group(pattern_group_index):
            print("context:", context)
            print("search_context:", search_context)
            print("re_match:", name_re_match.groups())
        match_offset += whole_match.index(name_re_match.group(pattern_group_index))
        name_match = {
            "name_string": name_re_match.group(pattern_group_index),
            "match_offset": match_offset,
            "context_match_term": context["match_term"],
            "context_match_string": context["match_string"],
            "context_match_offset": context["match_offset"],
            "context_pattern": search_context["name"],
        }
        # print("pattern_group_index:", pattern_group_index, "\tname_string:", name_match["name_string"])
        # print("index in while:", whole_match.index(name_re_match.group(pattern_group_index)))
        whole_match = whole_match[
                      whole_match.index(name_re_match.group(pattern_group_index)) + len(name_match["name_string"]):]
        match_offset += len(name_match["name_string"])
        add_name_match_if_new(name_match, name_matches)


def parse_person_name_in_text_matches(name_re_match, search_pattern, text, name_matches):
    match_offset = name_re_match.start()
    whole_match = name_re_match.group(0)
    for pattern_group_index in search_pattern["group_indices"]:
        if not name_re_match.group(pattern_group_index):
            print("context:", text)
            print("search_pattern:", search_pattern)
            print("re_match:", name_re_match.groups())
        match_offset += whole_match.index(name_re_match.group(pattern_group_index))
        name_match = {
            "name_string": name_re_match.group(pattern_group_index),
            "match_offset": match_offset,
            "pattern": search_pattern["type"],
        }
        # print("pattern_group_index:", pattern_group_index, "\tname_string:", name_match["name_string"])
        # print("index in while:", whole_match.index(name_re_match.group(pattern_group_index)))
        whole_match = whole_match[
                      whole_match.index(name_re_match.group(pattern_group_index)) + len(name_match["name_string"]):]
        match_offset += len(name_match["name_string"])
        add_name_match_if_new(name_match, name_matches)


if __name__ == "__main__":
    from fuzzy.fuzzy_patterns import dutch_person_name_patterns as name_patterns

    sample_text = "A'nthony van der Truyn en Adriaen Bosman, Makelaers tot Rotterdam, prefenteren, uyt de Hint te verkopen etfn curieufc Party opreckw ?al somfl'e Schalyen of Leyen."
    name_matches = find_person_names_in_context()
