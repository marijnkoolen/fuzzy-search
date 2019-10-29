import uuid
from itertools import combinations
from collections import defaultdict


class Keyword(object):

    def __init__(self, keyword, ngram_size=2, skip_size=2, early_threshold=3, late_threshold=3, within_range_threshold=3, ignorecase=False):
        if isinstance(keyword, str):
            keyword = {"keyword_string": keyword}
        self.name = keyword["keyword_string"]
        self.properties = keyword
        self.early_threshold = early_threshold
        self.late_threshold = len(self.name) - late_threshold - ngram_size
        self.within_range_threshold = within_range_threshold
        keyword_string = self.name
        self.ignorecase = ignorecase
        if ignorecase:
            keyword_string = keyword_string.lower()
        self.ngrams = [(ngram, offset) for ngram, offset in text2skipgrams(keyword_string, ngram_size=ngram_size, skip_size=skip_size)]
        self.ngram_index = defaultdict(list)
        for ngram, offset in self.ngrams:
            self.ngram_index[ngram] += [offset]
        self.index_ngrams()
        self.early_ngrams = {ngram: offset for ngram, offset in self.ngrams if offset < early_threshold}
        self.late_ngrams = {ngram: offset for ngram, offset in self.ngrams if offset > self.late_threshold}
        self.set_within_range()

    def index_ngrams(self):
        self.ngram_index = defaultdict(list)
        for ngram, offset in self.ngrams:
            self.ngram_index[ngram] += [offset]

    def has_ngram(self, ngram):
        return ngram in self.ngram_index.keys()

    def ngram_offsets(self, ngram):
        if not self.has_ngram(ngram):
            return None
        return self.ngram_index[ngram]

    def set_within_range(self):
        self.ngram_distance = {}
        for index1 in range(0, len(self.ngrams)-1):
            ngram1, offset1 = self.ngrams[index1]
            for index2 in range(index1+1, len(self.ngrams)):
                ngram2, offset2 = self.ngrams[index2]
                if offset2 - offset1 > self.within_range_threshold:
                    continue
                if (ngram1, ngram2) not in self.ngram_distance:
                    self.ngram_distance[(ngram1, ngram2)] = offset2 - offset1
                elif self.ngram_distance[(ngram1, ngram2)] > offset2 - offset1:
                    self.ngram_distance[(ngram1, ngram2)] = offset2 - offset1

    def within_range(self, ngram1, ngram2):
        if not self.has_ngram(ngram1) or not self.has_ngram(ngram2):
            return False
        elif (ngram1, ngram2) not in self.ngram_distance:
            return False
        elif self.ngram_distance[(ngram1, ngram2)] > self.within_range_threshold:
            return False
        else:
            return True

    def early_ngram(self, ngram):
        return ngram in self.early_ngrams

def insert_skips(window, ngram_combinations):
    for combination in ngram_combinations:
        prev_index = 0
        skip_gram = window[0]
        try:
            for index in combination:
                if index - prev_index > 1:
                    skip_gram += "_"
                skip_gram += window[index]
                prev_index = index
            yield skip_gram
        except IndexError:
            pass

def text2skipgrams(text, ngram_size=2, skip_size=2):
    indexes = [i for i in range(0,ngram_size+skip_size)]
    ngram_combinations = [combination for combination in combinations(indexes[1:], ngram_size-1)]
    for offset in range(0, len(text)-1):
        window = text[offset:offset+ngram_size+skip_size]
        for skip_gram in insert_skips(window, ngram_combinations):
            yield (skip_gram, offset)

class PersonName(object):

    def __init__(self, person_name_string):
        self.person_name_id = str(uuid.uuid4())
        self.name_string = person_name_string
        self.normalized_name = normalize_person_name(person_name_string)
        self.name_type = "person_name"
        self.spelling_variants = []
        self.distractor_terms = []

    def to_json(self):
        return {
            "person_name_id": self.person_name_id,
            "name_string": self.name_string,
            "normalize_name": self.normalized_name,
            "name_type": self.name_type,
            "spelling_variants": self.spelling_variants,
            "distractor_terms": self.distractor_terms
        }

    def add_spelling_variants(self, spelling_variants):
        for spelling_variant in spelling_variants:
            if spelling_variant not in self.spelling_variants:
                self.spelling_variants += [spelling_variant]

    def add_distractor_terms(self, distractor_terms):
        for distractor_term in distractor_terms:
            if distractor_term not in self.distractor_terms:
                self.distractor_terms += [distractor_term]

def normalize_person_name(person_name_string):
    normalized = person_name_string.title()
    infixes = [" van ", " de ", " der ", " du ", " le ", " la "]
    for infix in infixes:
        normalized = normalized.replace(infix.title(), infix.lower())
    return normalized



