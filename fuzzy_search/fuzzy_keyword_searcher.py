import re
from collections import defaultdict
from fuzzy_search.fuzzy_keyword import Keyword, text2skipgrams


#################################
# String manipulation functions #
#################################

def make_ngrams(term, n):
    term = "#{t}#".format(t=term)
    max_start = len(term) - n + 1
    return [term[start:start + n] for start in range(0, max_start)]


def strip_suffix(match):
    if len(match) <= 2:
        pass
    elif match[-2] in [" ", ","]:
        match = match[:-2]
    elif match[-2:] in [", ", ". ", "? ", ".f"]:
        match = match[:-2]
    elif match[-1] in [" ", ",", "."]:
        match = match[:-1]
    return match


#####################################
# Term similarity scoring functions #
#####################################


def score_ngram_overlap(term1, term2, ngram_size):
    term1_ngrams = make_ngrams(term1, ngram_size)
    term2_ngrams = make_ngrams(term2, ngram_size)
    overlap = 0
    for ngram in term1_ngrams:
        if ngram in term2_ngrams:
            term2_ngrams.pop(term2_ngrams.index(ngram))
            overlap += 1
    return overlap


def score_char_overlap_ratio(term1, term2):
    max_overlap = len(term1)
    overlap = score_char_overlap(term1, term2)
    return overlap / max_overlap


def score_ngram_overlap_ratio(term1, term2, ngram_size):
    max_overlap = len(make_ngrams(term1, ngram_size))
    overlap = score_ngram_overlap(term1, term2, ngram_size)
    return overlap / max_overlap


def score_levenshtein_distance_ratio(term1, term2):
    max_distance = max(len(term1), len(term2))
    distance = score_levenshtein_distance(term1, term2)
    return 1 - distance / max_distance


def score_levenshtein_distance(s1, s2, use_confuse=False):
    """Calculate Levenshtein distance between two string. Beyond the
    normal algorithm, a confusion matrix can be used to get non-binary
    scores for common confusion pairs.
    To use the confusion matrix, config the searcher with use_confuse=True"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                dist = confuse_distance(c1, c2) if use_confuse else 1
                distances_.append(dist + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def score_char_overlap(term1, term2):
    num_char_matches = 0
    for char in term2:
        if char in term1:
            term1 = term1.replace(char, "", 1)
            num_char_matches += 1
    return num_char_matches


#################################
# Helper functions #
#################################


def get_match_terms(match_term, candidate):
    if isinstance(candidate, str):
        return match_term, candidate
    else:
        return candidate["match_term"], candidate["match_string"]


def add_candidate_scores(info, candidate):
    if "char_match" in candidate:
        info["char_match"] = candidate["char_match"]
    if "ngram_match" in candidate:
        info["ngram_match"] = candidate["ngram_match"]
    if "levenshtein_distance" in candidate:
        info["levenshtein_distance"] = candidate["levenshtein_distance"]


def matches_overlap(match1, match2):
    """Determines wether two match strings overlap in the text"""
    if match1["match_offset"] <= match2["match_offset"]:
        return match1["match_offset"] + len(match1["match_string"]) > match2["match_offset"]
    elif match1["match_offset"] > match2["match_offset"]:
        return match2["match_offset"] + len(match2["match_string"]) > match1["match_offset"]


def rank_candidates(candidates, keyword, ngram_size=2):
    total_scores = []
    for candidate in candidates:
        if isinstance(candidate, str):
            match_string = candidate
        elif isinstance(candidate, dict):
            match_string = candidate["match_string"]
        score = {
            "candidate": candidate,
            "char": score_char_overlap_ratio(match_string, keyword),
            "ngram": score_ngram_overlap_ratio(match_string, keyword, ngram_size),
            "levenshtein": score_levenshtein_distance_ratio(match_string, keyword),
        }
        score["total"] = score["char"] + score["ngram"] + score["levenshtein"]
        total_scores += [score]
    return sorted(total_scores, key=lambda x: x["total"], reverse=True)


def get_best_keyword_offset(keyword_string, keyword_ngram_offsets, matches):
    # check if current keyword ngram closely follows last found keyword ngram
    last_keyword_offset = matches[-1][2]
    for offset in keyword_ngram_offsets:
        # skip offsets before last found offsets
        if offset < last_keyword_offset:
            continue
        # if offset is too far beyond last found ngram from
        elif offset > last_keyword_offset + 3:
            return None
        else:
            return offset
    # if no closely following offset can be found, return the first to start
    # a new candidate
    return keyword_ngram_offsets[0]


class FuzzyKeywordSearcher(object):

    def __init__(self, config):
        # default configuration
        self.char_match_threshold = 0.5
        self.ngram_threshold = 0.5
        self.levenshtein_threshold = 0.5
        self.perform_strip_suffix = True
        self.max_length_variance = 1
        self.use_word_boundaries = True
        self.ignorecase = False
        self.track_candidates = False
        self.use_confuse = False
        self.tracking_level = 4
        self.known_candidates = defaultdict(dict)
        self.distractor_terms = defaultdict(list)
        self.ngram_size = 2
        self.skip_size = 2
        # non-default configuration
        self.configure(config)
        self.variant_map = defaultdict(dict)
        self.has_variant = defaultdict(dict)
        self.variant_ngram_index = defaultdict(dict)

    def configure(self, config):
        if "char_match_threshold" in config:
            self.char_match_threshold = config["char_match_threshold"]
        if "ngram_threshold" in config:
            self.ngram_threshold = config["ngram_threshold"]
        if "levenshtein_threshold" in config:
            self.levenshtein_threshold = config["levenshtein_threshold"]
        if "max_length_variance" in config:
            self.max_length_variance = config["max_length_variance"]
        if "use_word_boundaries" in config:
            self.use_word_boundaries = config["use_word_boundaries"]
        if "ignorecase" in config:
            self.ignorecase = config["ignorecase"]
        if "track_candidates" in config:
            self.track_candidates = config["track_candidates"]
        if "use_confuse" in config:
            self.use_confuse = config["use_confuse"]
        if "ngram_size" in config:
            self.ngram_size = config["ngram_size"]
        if "skip_size" in config:
            self.skip_size = config["skip_size"]

    def enable_strip_suffix(self):
        self.perform_strip_suffix = True

    def disable_strip_suffix(self):
        self.perform_strip_suffix = False

    ##############################
    # Keyword indexing functions #
    ##############################

    def index_keywords(self, keywords, ignorecase=None):
        """Index keywords as ngrams for quick lookup of keywords that match an ngram.
        The early_ngram_index only indexes the ngrams starting with one of the first
        three characters, to quickly lookup if a given ngram could be the start of a
        keyword mention.
        The late_ngram_index only indexes the ngrams ending with one of the final three
        characters, to quickly lookup if a given ngram could be the end of a keyword
        mention."""
        if not ignorecase:
            ignorecase = self.ignorecase
        self.keyword_index = make_keyword_index(keywords, ngram_size=self.ngram_size, skip_size=self.skip_size,
                                                ignorecase=ignorecase)
        self.ngram_index = index_keyword_ngrams(self.keyword_index)
        self.early_ngram_index = index_early_keyword_ngrams(self.keyword_index)
        self.late_ngram_index = index_late_keyword_ngrams(self.keyword_index)
        # print("set late_ngram_index:", self.late_ngram_index)

    def index_spelling_variants(self, keyword_variants):
        """Index spelling variants of an indexed keyword, to see if a keyword mention is
        closer in spelling to a known variant than to the keyword itself."""
        for keyword in keyword_variants:
            if keyword not in self.keyword_index:
                raise KeyError("Unknown keyword:", keyword)
            for variant in keyword_variants[keyword]:
                self.index_spelling_variant(keyword, variant)

    def index_spelling_variant(self, keyword, variant):
        variant_keyword = Keyword(variant, ngram_size=self.ngram_size, skip_size=self.skip_size,
                                  ignorecase=self.ignorecase)
        self.variant_map[variant][keyword] = variant_keyword
        self.has_variant[keyword][variant] = variant_keyword
        for ngram, offset in variant_keyword.early_ngrams.items():
            self.variant_ngram_index[ngram][variant] = offset

    def index_distractor_terms(self, keyword_distractor_terms):
        """input: a dictionary with keywords as keys and lists of distractor terms as values."""
        for keyword in keyword_distractor_terms:
            self.distractor_terms[keyword] = keyword_distractor_terms[keyword]
        self.update_known_candidates(keyword_distractor_terms)

    def update_known_candidates(self, distractor_terms):
        for match_string in self.known_candidates:
            for match_term in self.known_candidates[match_string]:
                self.update_known_candidate(match_string, match_term, distractor_terms)

    def update_known_candidate(self, match_string, match_term, distractor_terms):
        # if status is already reject, skip test
        if self.known_candidates[match_string][match_term]["status"] == "reject":
            return False
        # if status is accept, test match_string against distractors
        if match_term in distractor_terms:
            self.test_known_candidate_status(match_string, match_term, distractor_terms[match_term])
        if match_term not in self.variant_map:
            return False
        for keyword in self.variant_map[match_term]:
            if keyword not in distractor_terms:
                continue
            self.test_known_candidate_status(match_string, match_term, distractor_terms[keyword])

    def test_known_candidate_status(self, match_string, match_term, distractors):
        accept_distance = score_levenshtein_distance(match_string, match_term)
        for distractor in distractors:
            reject_distance = score_levenshtein_distance(match_string, distractor)
            if reject_distance < accept_distance:
                self.known_candidates[match_string][match_term]["status"] = "reject"
                return True
        return False

    #################################
    # Helper functions #
    #################################

    def track_level(self, filter_type):
        """Function to determine if a found candidate should be stored for later
        matching to skip expensive filtering operations on previously seen candidates."""
        if not self.track_candidates:
            return False
        if filter_type == "char_match" and self.tracking_level <= 1:
            return True
        if filter_type == "ngram_match" and self.tracking_level <= 2:
            return True
        if filter_type == "levenshtein_distance" and self.tracking_level <= 3:
            return True

    def add_known_candidate(self, candidate, status):
        if not self.track_candidates:
            return None
        match_string = candidate["match_string"]
        match_term = candidate["match_term"]
        self.known_candidates[match_string][match_term] = {
            "status": status,
        }
        add_candidate_scores(self.known_candidates[match_string][match_term], candidate)

    def get_candidate_score(self, match_term, candidate, filter_type, ngram_size):
        term1, term2 = get_match_terms(match_term, candidate)
        if filter_type == "char_match":
            return score_char_overlap_ratio(term1, term2)
        if filter_type == "ngram_match":
            return score_ngram_overlap_ratio(term1, term2, ngram_size)
        if filter_type == "levenshtein_distance":
            return score_levenshtein_distance_ratio(term1, term2)

    def above_threshold(self, score, filter_type):
        if filter_type == "char_match":
            return score >= self.char_match_threshold
        if filter_type == "ngram_match":
            return score >= self.ngram_threshold
        if filter_type == "levenshtein_distance":
            return score >= self.levenshtein_threshold

    def is_distractor_candidate(self, candidate):
        if "match_keyword" in candidate:
            keyword = candidate["match_keyword"]
        else:
            keyword = candidate["match_term"]
        # if keyword has no distractor_terms, candidate is not distractor term
        if keyword not in self.distractor_terms:
            return False
        keyword_distance = score_levenshtein_distance(candidate["match_term"], candidate["match_string"])
        for distractor_term in self.distractor_terms[keyword]:
            distractor_distance = score_levenshtein_distance(distractor_term, candidate["match_string"])
            if distractor_distance < keyword_distance:
                return True
        return False

    def already_found_match(self, keyword_string, match_offset):
        if keyword_string not in self.found:
            return False
        if match_offset in self.found[keyword_string]:
            return True
        return False

    def get_ngram_entries_new(self, ngram_string):
        if ngram_string in self.ngram_index:
            for keyword, offsets in self.ngram_index[ngram_string].items():
                yield keyword, offsets
        if self.include_variants and ngram_string in self.variant_ngram_index:
            for variant, offsets in self.variant_ngram_index[ngram_string].items():
                yield variant, offsets

    def get_ngram_entries(self, ngram_string):
        if ngram_string in self.early_ngram_index:
            for keyword, offset in self.early_ngram_index[ngram_string].items():
                yield keyword, offset
        if self.include_variants and ngram_string in self.variant_ngram_index:
            for variant, offset in self.variant_ngram_index[ngram_string].items():
                yield variant, offset

    def get_match_offset(self, text, ngram_offset, keyword_ngram_offset):
        match_offset = ngram_offset - keyword_ngram_offset
        if self.use_word_boundaries:
            match_offset = adjust_match_offset(match_offset, text)
        return match_offset

    def get_match_end(self, text, match_offset, keyword_string):
        match_end = match_offset + len(keyword_string)
        if self.use_word_boundaries:
            match_end = adjust_match_end(match_end, text)
        return match_end

    def get_match_details(self, text, keyword_string, keyword_ngram_offset, ngram_offset):
        match_offset = self.get_match_offset(text, ngram_offset, keyword_ngram_offset)
        if self.already_found_match(keyword_string, match_offset):
            return None
        if match_offset < 0:
            return None
        match_end = self.get_match_end(text, match_offset, keyword_string)
        if match_end < 0:
            return None
        match_string = text[match_offset:match_end]
        if self.perform_strip_suffix:
            match_string = strip_suffix(match_string)
        return self.make_ngram_match(keyword_string, match_string, match_offset)

    def make_ngram_match(self, keyword_string, match_string, match_offset):
        match_keyword = keyword_string
        if keyword_string not in self.keyword_index:
            match_keyword = list(self.variant_map[keyword_string].keys())
        if len(match_keyword) == 1:
            match_keyword = match_keyword[0]
        return {
            "match_keyword": match_keyword,
            "match_term": keyword_string,
            "match_string": match_string,
            "match_offset": match_offset
        }

    def get_known_candidate(self, candidate):
        if self.is_known_candidate(candidate):
            return self.known_candidates[candidate["match_string"]][candidate["match_term"]]
        elif not self.track_candidates or not self.is_known_candidate(candidate):
            return None
        else:
            return self.known_candidates[candidate["match_string"]][candidate["match_term"]]

    def is_rejected_candidate(self, candidate):
        if not self.track_candidates or not self.is_known_candidate(candidate):
            return False
        else:
            return self.get_known_candidate_status(candidate) == "reject"

    def set_known_candidate_status(self):
        for status in self.candidates:
            for candidate in self.candidates[status]:
                match_string = candidate["match_string"]
                keyword = candidate["match_term"]
                if self.is_known_candidate(candidate):
                    self.known_candidates[match_string][keyword]["status"] = status
                    continue
                info = {
                    "status": status,
                }
                add_candidate_scores(info, candidate)
                self.known_candidates[match_string][keyword] = info

    def is_known_candidate(self, candidate):
        if candidate["match_string"] not in self.known_candidates:
            return False
        elif candidate["match_term"] not in self.known_candidates[candidate["match_string"]]:
            return False
        else:
            return True

    def get_known_candidate_status(self, candidate):
        """Determines whether a combination of match string and
        match term has been seen before and has a known status."""
        return self.known_candidates[candidate["match_string"]][candidate["match_term"]]["status"]

    #################################
    # Candidate filtering functions #
    #################################

    def filter_all_candidates(self, ngram_size):
        if self.include_variants:
            self.filter_variant_candidates()
        self.filter_candidates(filter_type="char_match")
        self.filter_candidates(filter_type="ngram_match", ngram_size=ngram_size)
        self.filter_candidates(filter_type="levenshtein_distance")
        self.filter_overlapping_matches()
        if self.filter_distractors:
            self.filter_distractor_candidates()
        else:
            self.candidates["accept"] += self.candidates["filter"]
            self.candidates["filter"] = []

    def filter_candidates(self, filter_type, match_term=None, ngram_size=2):
        candidates = self.candidates["filter"]
        filtered = []
        for candidate in self.candidates["filter"]:
            if filter_type not in candidate:
                score = self.get_candidate_score(match_term, candidate, filter_type, ngram_size)
                candidate[filter_type] = score
            if self.above_threshold(candidate[filter_type], filter_type):
                filtered += [candidate]
            else:
                self.candidates["reject"] += [candidate]
                if self.track_level(filter_type):
                    self.add_known_candidate(candidate, "reject")
        self.candidates["filter"] = filtered

    def filter_variant_candidates(self):
        best_candidate = {}
        best_distance = {}
        for candidate in self.candidates["filter"]:
            if "levenshtein_distance" not in candidate:
                levenshtein_distance = score_levenshtein_distance_ratio(candidate["match_term"],
                                                                        candidate["match_string"])
                candidate["levenshtein_distance"] = levenshtein_distance
            offset = candidate["match_offset"]
            if candidate["match_term"] in self.keyword_index:
                keywords = [candidate["match_term"]]
            else:
                keywords = self.variant_map[candidate["match_term"]].keys()
            for keyword in keywords:
                if candidate["match_term"] not in self.keyword_index:
                    candidate["match_keyword"] = keyword
                if (keyword, offset) not in best_candidate:
                    best_candidate[(keyword, offset)] = candidate
                    best_distance[(keyword, offset)] = candidate["levenshtein_distance"]
                elif candidate["levenshtein_distance"] > best_distance[(keyword, offset)]:
                    self.add_known_candidate(best_candidate[(keyword, offset)], "reject")
                    best_candidate[(keyword, offset)] = candidate
                    best_distance[(keyword, offset)] = candidate["levenshtein_distance"]
                else:
                    self.add_known_candidate(candidate, "reject")
        self.candidates["filter"] = list(best_candidate.values())

    def filter_distractor_candidates(self):
        for candidate in self.candidates["filter"]:
            if self.is_distractor_candidate(candidate):
                self.candidates["reject"] += [candidate]
                self.add_known_candidate(candidate, "reject")
            else:
                self.candidates["accept"] += [candidate]
        self.candidates["filter"] = []

    def filter_overlapping_matches(self):
        self.candidates["filter"].sort(key=lambda x: x["match_offset"])
        filtered = []
        for match in self.candidates["filter"]:
            if len(filtered) == 0:
                # print("Adding match", match)
                filtered += [match]
                continue
            prev_offset = filtered[-1]["match_offset"]
            curr_offset = match["match_offset"]
            if prev_offset + len(filtered[-1]["match_term"]) < curr_offset:
                filtered += [match]
                continue
            elif match["levenshtein_distance"] > filtered[-1]["levenshtein_distance"]:
                # this match overlaps with previous and is better, so replace
                filtered[-1] = match
            else:
                # this match overlaps with previous and is worse, so skip
                continue
        self.candidates["filter"] = filtered

    ##########################################
    # Functions for finding patterns in text #
    ##########################################

    def find_initial_char_matches(self, text, term, max_length_variance=None):
        if not max_length_variance:
            max_length_variance = self.max_length_variance
        initial = term[0]
        length_range = {"min": len(term[1:]) - max_length_variance, "max": len(term[1:]) + max_length_variance}
        if initial in ["[", "]", "*", "(", ")", "."]:
            initial = "\\" + initial
        pattern = initial + ".{" + str(length_range["min"]) + "," + str(length_range["max"]) + "}"
        if self.use_word_boundaries:
            pattern = r"\b" + pattern + r"\b"
        try:
            return [create_regex_match(re_match, term) for re_match in re.finditer(pattern, text)]
        except TypeError:
            print("\n\nERROR\n\n")
            print("text:", text)
            print("term:", term)
            print("pattern:", pattern)
            raise

    def find_initial_char_candidates(self, text, term):
        for match in self.find_initial_char_matches(text, term):
            if self.perform_strip_suffix:
                match["match_string"] = strip_suffix(match["match_string"])
            yield match

    def find_ngram_candidates_new(self, text, ignorecase=None):
        self.include_variants = False
        candidates = []
        self.found = defaultdict(dict)
        self.open_candidates = defaultdict(dict)
        self.closed_candidates = []
        if ignorecase or (ignorecase is None and self.ignorecase):
            text = text.lower()
        for ngram_string, ngram_offset in text2skipgrams(text, ngram_size=self.ngram_size, skip_size=self.skip_size):
            self.find_ngram_matches_new(ngram_string, ngram_offset, text)
        open_candidates = list(self.open_candidates.keys())
        for keyword_string in open_candidates:
            offsets = list(self.open_candidates[keyword_string].keys())
            for start_offset in offsets:
                self.remove_candidate(keyword_string, start_offset)
        candidates = make_closed_candidate_matches(text, self.closed_candidates)
        return candidates

    def find_ngram_matches_new(self, ngram_string, ngram_offset, text):
        matches = []
        for keyword_string, keyword_ngram_offsets in self.get_ngram_entries_new(ngram_string):
            # print(keyword_string, keyword_ngram_offsets, ngram_string, ngram_offset)
            if keyword_string in self.open_candidates:
                if self.test_open_candidate(ngram_string, ngram_offset, keyword_string, keyword_ngram_offsets):
                    continue
            # if ngram does not match open candidate, check if matches the start of
            # the keyword string
            if keyword_ngram_offsets[0] < 3:
                # add ngram as start of new open candidate
                self.open_candidates[keyword_string][ngram_offset] = [
                    (ngram_string, ngram_offset, keyword_ngram_offsets[0])]
                # print("starting candidate:", self.open_candidates[keyword_string][ngram_offset], ngram_string, ngram_offset)
            else:
                pass

    def test_open_candidate(self, ngram_string, ngram_offset, keyword_string, keyword_ngram_offsets):
        matches_open_candidate = False
        remove_offsets = []
        for start_offset in self.open_candidates[keyword_string]:
            matches = self.open_candidates[keyword_string][start_offset]
            if ngram_offset > matches[-1][1] + 3:
                remove_offsets += [start_offset]
            else:
                best_keyword_offset = get_best_keyword_offset(keyword_string, keyword_ngram_offsets, matches)
                if best_keyword_offset is not None and best_keyword_offset >= matches[-1][2]:
                    # if best offset closely follows last match, add current
                    # as next match
                    matches += [(ngram_string, ngram_offset, best_keyword_offset)]
                    matches_open_candidate = True
                # print(keyword_string, keyword_ngram_offsets, ngram_string, ngram_offset)
                # print("continuing candidate:", keyword_string, self.open_candidates[keyword_string], ngram_string, ngram_offset)
        for start_offset in remove_offsets:
            self.remove_candidate(keyword_string, start_offset)
        return matches_open_candidate

    def remove_candidate(self, keyword_string, start_offset):
        matches = self.open_candidates[keyword_string][start_offset]
        last_ngram = matches[-1][0]
        last_offset = matches[-1][2]
        # print("last_gram:", last_ngram)
        # print("late_ngram_index:", self.late_ngram_index[last_ngram])
        if keyword_string in self.late_ngram_index[last_ngram]:
            if self.late_ngram_index[last_ngram][keyword_string] == last_offset:
                self.close_candidate(keyword_string, start_offset)
        # print("removing candidate:", keyword_string, self.open_candidates[keyword_string])
        del self.open_candidates[keyword_string][start_offset]
        if len(self.open_candidates[keyword_string].keys()) == 0:
            del self.open_candidates[keyword_string]

    def close_candidate(self, keyword_string, start_offset):
        first = self.open_candidates[keyword_string][start_offset][0]
        text_offset = first[1]
        keyword_offset = first[2]
        last = self.open_candidates[keyword_string][start_offset][-1]
        text_end = last[1] + len(keyword_string) - last[2] + 1
        # print("text_start:", text_offset, "text_end:", text_end, "keyword length:", len(keyword_string))
        self.closed_candidates += [(keyword_string, text_offset, text_end, keyword_offset)]
        # print(self.closed_candidates)
        # print("closing candidate:", keyword_string, self.open_candidates[keyword_string])

    def find_ngram_candidates(self, text, ignorecase=None):
        candidates = []
        self.found = defaultdict(dict)
        if ignorecase or (ignorecase is None and self.ignorecase):
            text = text.lower()
        for ngram_string, ngram_offset in text2skipgrams(text, ngram_size=self.ngram_size, skip_size=self.skip_size):
            for match in self.find_ngram_matches(ngram_string, ngram_offset, text):
                self.found[match["match_term"]][match["match_offset"]] = 1
                candidates += [match]
        return candidates

    def find_ngram_matches(self, ngram_string, ngram_offset, text):
        matches = []
        for keyword_string, keyword_ngram_offset in self.get_ngram_entries(ngram_string):
            try:
                match = self.get_match_details(text, keyword_string, keyword_ngram_offset, ngram_offset)
            except IndexError:
                print(len(text[keyword_ngram_offset:]), text[keyword_ngram_offset:])
                print(keyword_string, keyword_ngram_offset, ngram_string, ngram_offset)
                raise
            if not match:
                continue
            if self.is_known_candidate(match):
                known_candidate = self.get_known_candidate(match)
                match["status"] = known_candidate["status"]
                add_candidate_scores(match, known_candidate)
            # print("\t", match)
            # print()
            yield match

    def find_all_candidates(self, text, keyword):
        self.candidates = defaultdict(list)
        if self.match_initial_char:
            candidates = self.find_initial_char_candidates(text, keyword)
        else:
            candidates = self.find_ngram_candidates(text)
        self.candidates["filter"] = [candidate for candidate in candidates if not self.is_rejected_candidate(candidate)]

    def find_candidates(self, text, keyword=None, ngram_size=2, use_word_boundaries=None, match_initial_char=False,
                        include_variants=False, filter_distractors=False):
        self.include_variants = include_variants
        self.filter_distractors = filter_distractors
        self.match_initial_char = match_initial_char
        if isinstance(use_word_boundaries, bool):
            self.use_word_boundaries = use_word_boundaries
        self.find_all_candidates(text, keyword)
        self.filter_all_candidates(ngram_size)
        if self.track_candidates:
            self.set_known_candidate_status()
        return self.candidates["accept"]

    def find_candidates_new(self, text, keyword=None, ngram_size=2, use_word_boundaries=None, match_initial_char=False,
                            include_variants=False, filter_distractors=False):
        self.include_variants = include_variants
        self.filter_distractors = filter_distractors
        self.match_initial_char = match_initial_char
        if isinstance(use_word_boundaries, bool):
            self.use_word_boundaries = use_word_boundaries
        candidates = self.find_ngram_candidates_new(text)
        self.candidates = defaultdict(list)
        self.candidates["filter"] = [candidate for candidate in candidates if not self.is_rejected_candidate(candidate)]
        self.filter_all_candidates(ngram_size)
        if self.track_candidates:
            self.set_known_candidate_status()
        return self.candidates["accept"]

    def find_close_distance_keywords(self, keyword_list, max_distance_ratio=0.3):
        close_distance_keywords = defaultdict(list)
        for index, keyword1 in enumerate(keyword_list):
            string1 = get_keyword_string(keyword1).lower()
            for keyword2 in keyword_list[index + 1:]:
                string2 = get_keyword_string(keyword2).lower()
                # TODO:
                # - consider use of initials for first or last keyword, e.g. A. Doelman
                # - consider use of only last keyword, e.g. Doelman
                # skip comparison if:
                # - keywords are very different in length
                if abs(len(string1) - len(string2)) > 3: continue
                # - keywords have low overlap in characters
                char_overlap = self.score_char_overlap(string1, string2)
                # print(string1, string2, char_overlap, char_overlap/len(string1))
                if char_overlap / len(string1) < 0.5: continue
                distance = self.score_levenshtein_distance(string1, string2)
                # print(string1, string2, distance, distance / len(string1), distance/len(string2))
                if distance < 10 and (
                        distance / len(string1) < max_distance_ratio or distance / len(string2) < max_distance_ratio):
                    close_distance_keywords[keyword1].append(keyword2)
                    close_distance_keywords[keyword2].append(keyword1)
        return close_distance_keywords

    def find_closer_terms(self, candidate, keyword, close_terms):
        closer_terms = {}
        keyword_distance = self.score_levenshtein_distance(keyword, candidate)
        # print("candidate:", candidate, "\tkeyword:", keyword)
        # print("keyword_distance", keyword_distance)
        for close_term in close_terms:
            close_term_distance = self.score_levenshtein_distance(close_term, candidate)
            # print("close_term:", close_term, "\tdistance:", close_term_distance)
            if close_term_distance < keyword_distance:
                closer_terms[close_term] = close_term_distance
        return sorted(closer_terms, key=lambda closer_terms: closer_terms[1])


####################
# Helper functions #
####################

def make_closed_candidate_matches(text, candidates):
    return [make_closed_candidate_match(text, candidate) for candidate in candidates]


def make_closed_candidate_match(text, candidate):
    return {
        "match_keyword": candidate[0],
        "match_term": candidate[0],
        "match_string": text[candidate[1]:candidate[2]],
        "match_offset": candidate[1],
    }


def adjust_match_offset(match_offset, text):
    if match_offset <= 0:
        return 0
        # return match_offset
    # if match starts with single whitespace, skip first char
    elif text[match_offset] == " " and text[match_offset + 1 != " "]:
        return match_offset + 1
    # if match is preceded by a non alpha-numeric character, keep match offset
    elif re.match(r"\W", text[match_offset - 1]):
        return match_offset
    # if word boundary is one char before offset, adjust offset to preceding char
    elif re.match(r"\W", text[match_offset - 2]) or match_offset == 1:
        return match_offset - 1
    # if match is in middle of word, set impossible offset so match is ignored
    else:
        return -1


def adjust_match_end(match_end, text):
    # if match_end is more than two characters beyond end of text, match is invalid
    if match_end > len(text) + 2:
        return -1
    # if match_end is end of text or at most two characters more, use text end
    elif match_end >= len(text):
        return len(text)
    elif text[match_end - 2] == " ":
        return match_end - 2
    elif text[match_end - 1] == " ":
        return match_end - 1
    elif text[match_end] == " " or match_end + 1 == len(text):
        return match_end
    elif re.match(r"\W", text[match_end]):
        return match_end
    elif text[match_end + 1] == " " or match_end + 2 == len(text):
        return match_end + 1
    elif re.match(r"\W", text[match_end + 1]):
        return match_end + 1
    elif text[match_end + 2] == " " or match_end + 3 == len(text):
        return match_end + 2
    elif re.match(r"\W", text[match_end + 2]):
        return match_end + 2
    else:
        return -1


def create_regex_match(re_match, term):
    return {
        "match_term": term,
        "match_string": re_match.group(0),
        "match_offset": re_match.start()
    }


def make_keyword(keyword, ngram_size=2, skip_size=2, ignorecase=False):
    if isinstance(keyword, Keyword):
        return keyword
    elif isinstance(keyword, dict) or isinstance(keyword, str):
        return Keyword(keyword, ngram_size=ngram_size, skip_size=skip_size, ignorecase=ignorecase)


def make_keyword_index(keywords, ngram_size=2, skip_size=2, ignorecase=True):
    if not isinstance(keywords, list):
        keywords = [keywords]
    keywords = [make_keyword(keyword, ngram_size=ngram_size, skip_size=skip_size, ignorecase=ignorecase) for keyword in
                keywords]
    return {
        keyword.name: keyword for keyword in keywords
    }


def index_keyword_ngrams(keyword_index):
    ngram_index = defaultdict(dict)
    for keyword_string in keyword_index:
        for ngram, offset in keyword_index[keyword_string].ngrams:
            if keyword_string not in ngram_index[ngram]:
                ngram_index[ngram][keyword_string] = []
            ngram_index[ngram][keyword_string] += [offset]
    return ngram_index


def index_early_keyword_ngrams(keyword_index):
    early_ngram_index = defaultdict(dict)
    for keyword_string in keyword_index:
        for ngram, offset in keyword_index[keyword_string].early_ngrams.items():
            early_ngram_index[ngram][keyword_string] = offset
    return early_ngram_index


def index_late_keyword_ngrams(keyword_index):
    late_ngram_index = defaultdict(dict)
    for keyword_string in keyword_index:
        for ngram, offset in keyword_index[keyword_string].late_ngrams.items():
            late_ngram_index[ngram][keyword_string] = offset
    # print("returning late_ngram_index:", late_ngram_index)
    return late_ngram_index


def get_keyword_string(keyword):
    if isinstance(keyword, str):
        return keyword
    elif isinstance(keyword, Keyword):
        return keyword.name
    elif isinstance(keyword, dict) and "keyword_string" in keyword:
        return keyword["keyword_string"]
    else:
        return None


def is_confuse_pair(c1, c2):
    if (c1, c2) in pairs:
        return True
    elif (c2, c1) in pairs:
        return True
    else:
        return False


def confuse_distance(c1, c2):
    if (c1, c2) in pairs:
        return pairs[(c1, c2)]
    elif (c2, c1) in pairs:
        return pairs[(c2, c1)]
    else:
        return 1


pairs = {
    # common variants lower case
    ('s', 'f'): 0.5,
    ('c', 'k'): 0.5,
    # common confusions lower case
    ('j', 'i'): 0.5,
    ('r', 'i'): 0.5,
    ('r', 't'): 0.5,
    ('r', 'n'): 0.5,
    ('e', 'c'): 0.5,
    ('a', 'e'): 0.5,
    ('a', 'c'): 0.5,
    ('o', 'c'): 0.5,
    ('y', 'i'): 0.5,
    ('l', 'i'): 0.5,
    # common variants upper case
    ('C', 'K'): 0.5,
    # common confusions upper case
    ('I', 'l'): 0.5,
    ('J', 'T'): 0.5,
    ('P', 'F'): 0.5,
    ('P', 'T'): 0.5,
    ('T', 'F'): 0.5,
    ('I', 'L'): 0.5,
    ('B', '8'): 0.5,
    # lower case vs. upper case
    ('a', 'A'): 0.1,
    ('b', 'B'): 0.1,
    ('c', 'C'): 0.1,
    ('d', 'D'): 0.1,
    ('e', 'E'): 0.1,
    ('f', 'F'): 0.1,
    ('g', 'G'): 0.1,
    ('h', 'H'): 0.1,
    ('i', 'I'): 0.1,
    ('j', 'J'): 0.1,
    ('k', 'K'): 0.1,
    ('l', 'L'): 0.1,
    ('m', 'M'): 0.1,
    ('n', 'N'): 0.1,
    ('o', 'O'): 0.1,
    ('p', 'P'): 0.1,
    ('q', 'Q'): 0.1,
    ('r', 'R'): 0.1,
    ('s', 'S'): 0.1,
    ('t', 'T'): 0.1,
    ('u', 'U'): 0.1,
    ('v', 'V'): 0.1,
    ('w', 'W'): 0.1,
    ('x', 'X'): 0.1,
    ('y', 'Y'): 0.1,
    ('z', 'Z'): 0.1,
    # diacritic vs. no diacritic
    ('e', 'é'): 0.1,
    ('e', 'ë'): 0.1,
    ('e', 'è'): 0.1,
    ('a', 'ä'): 0.1,
    ('a', 'á'): 0.1,
    ('a', 'à'): 0.1,
    ('i', 'ï'): 0.1,
    ('i', 'í'): 0.1,
    ('i', 'ì'): 0.1,
    ('o', 'ó'): 0.1,
    ('o', 'ö'): 0.1,
    ('o', 'ò'): 0.1,
    ('u', 'ú'): 0.1,
    ('u', 'ü'): 0.1,
    ('u', 'ù'): 0.1,
}
