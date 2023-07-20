import re
from typing import Dict, List, Set
from collections import defaultdict

from fuzzy_search.phrase.phrase import Phrase
from fuzzy_search.phrase.phrase_model import PhraseModel
from fuzzy_search.match.phrase_match import PhraseMatch


def index_known_word_offsets(exact_matches: List[PhraseMatch]) -> Dict[int, Dict[str, any]]:
    exact_match_offset: Dict[int, Set[PhraseMatch]] = defaultdict(set)
    known_word_offset: Dict[int, Dict[str, any]] = defaultdict(dict)
    for exact_match in exact_matches:
        match_words = re.split(r"\W+", exact_match.string)
        # print("exact match:", exact_match)
        text_offset = exact_match.offset
        match_offset = 0
        # print("text_offset:", text_offset, "\tmatch_offset:", match_offset)
        for match_word in match_words:
            start = text_offset + match_offset + exact_match.string[match_offset:].index(match_word)
            # print("match_word:", match_word, "\tstart:", start)
            end = start + len(match_word)
            if start not in known_word_offset:
                known_word = {
                    "word": match_word,
                    "start": start,
                    "end": end,
                    "match_phrases": {exact_match.string}
                }
                # print(known_word)
                known_word_offset[start] = known_word
            known_word_offset[start]["match_phrases"].add(exact_match.string)
            match_word_offset = match_offset + exact_match.string[match_offset:].index(match_word)
            # print("text_offset:", text_offset, "\tmatch_offset:", match_offset)
        exact_match_offset[exact_match.offset].add(exact_match)
    return known_word_offset


def search_exact_phrases(phrase_model: PhraseModel, text: Dict[str, str],
                         ignorecase: bool = False, use_word_boundaries: bool = True,
                         include_variants: bool = False, debug: int = 0):
    if use_word_boundaries:
        # print('searching with word boundaries')
        return search_exact_phrases_with_word_boundaries(phrase_model, text, ignorecase=ignorecase,
                                                         include_variants=include_variants, debug=debug)
    else:
        return search_exact_phrases_without_word_boundaries(phrase_model, text, ignorecase=ignorecase,
                                                            include_variants=include_variants, debug=debug)


def add_exact_match_score(match: PhraseMatch) -> PhraseMatch:
    match.character_overlap = 1.0
    match.ngram_overlap = 1.0
    match.levenshtein_similarity = 1.0
    return match


def search_exact_phrases_with_word_boundaries(phrase_model: PhraseModel, text: Dict[str, str],
                                              ignorecase: bool = False, include_variants: bool = False,
                                              debug: int = 0):
    for word in re.finditer(r"\w+", text["text"]):
        if debug > 0:
            print('search_exact_phrases_with_word_boundaries - word:', word)
            print('search_exact_phrases_with_word_boundaries - word in phrase_model:', word.group(0) not in phrase_model.word_in_phrase)
        if word.group(0) not in phrase_model.word_in_phrase:
            continue
        if debug > 0:
            print("\tsearch_exact_phrases_with_word_boundaries - matching word:", word)
        for phrase_string in phrase_model.first_word_in_phrase[word.group(0)]:
            phrase_word_offset = phrase_model.first_word_in_phrase[word.group(0)][phrase_string]
            phrase_start = word.start() - phrase_word_offset
            phrase_end = phrase_start + len(phrase_string)
            if debug > 0:
                print('\tsearch_exact_phrases_with_word_boundaries - start, end, string:', phrase_start, phrase_end,
                      phrase_string)
                print('\tsearch_exact_phrases_with_word_boundaries - text start, end:',
                      text["text"][phrase_start:phrase_end])
                print('\tsearch_exact_phrases_with_word_boundaries - same?:',
                      text["text"][phrase_start:phrase_end] == phrase_string)

            if text["text"][phrase_start:phrase_end] == phrase_string:
                if phrase_start > 0 and re.match(r'\w', text["text"][phrase_start - 1]):
                    if debug > 0:
                        print('\tsearch_exact_phrases_with_word_boundaries - match word is not at start word boundary')
                    continue
                if phrase_end < len(text['text']) - 1 and re.match(r'\w', text['text'][phrase_end]):
                    if debug > 0:
                        print('\tsearch_exact_phrases_with_word_boundaries - match word is not at end word boundary')
                    continue

                if debug > 0:
                    print('\tsearch_exact_phrases_with_word_boundaries - phrase_string, phrase_type:', phrase_string,
                      phrase_model.phrase_type[phrase_string])
                if "phrase" in phrase_model.phrase_type[phrase_string]:
                    if debug > 0:
                        print('\tsearch_exact_phrases_with_word_boundaries - match word is phrase')
                    phrase = phrase_model.phrase_index[phrase_string]
                    match = PhraseMatch(phrase, phrase, phrase_string, phrase_start, text_id=text["id"],
                                        ignorecase=ignorecase)
                    yield add_exact_match_score(match)
                    if debug > 0:
                        print("\tsearch_exact_phrases_with_word_boundaries - the matching phrase:", phrase)
                elif "variant" in phrase_model.phrase_type[phrase_string] and include_variants:
                    if debug > 0:
                        print('\tsearch_exact_phrases_with_word_boundaries - match word is variant')
                    variant_phrase = phrase_model.variant_index[phrase_string]
                    main_phrase_string = phrase_model.is_variant_of[phrase_string]
                    main_phrase = phrase_model.phrase_index[main_phrase_string]
                    match = PhraseMatch(main_phrase, variant_phrase, phrase_string, phrase_start,
                                        text_id=text["id"], ignorecase=ignorecase)
                    yield add_exact_match_score(match)


def search_exact_phrases_without_word_boundaries(phrase_model: PhraseModel, text: Dict[str, str],
                                                 ignorecase: bool = False,
                                                 include_variants: bool = False, debug: int = 0):
    for phrase_string in phrase_model.phrase_index:
        phrase = phrase_model.phrase_index[phrase_string]
        if debug > 0:
            print('search_exact_phrases_without_word_boundaries - phrase_string:', phrase_string)
        for match in re.finditer(phrase.exact_string, text["text"]):
            if debug > 0:
                print('search_exact_phrases_without_word_boundaries - match:', match)
            phrase = phrase_model.phrase_index[phrase_string]
            match = PhraseMatch(phrase, phrase, phrase_string, match.start(), text_id=text["id"],
                                ignorecase=ignorecase)
            yield add_exact_match_score(match)
    if include_variants:
        for phrase_string in phrase_model.variant_index:
            if debug > 0:
                print('search_exact_phrases_without_word_boundaries - phrase_string:', phrase_string)
            variant_phrase = phrase_model.variant_index[phrase_string]
            for match in re.finditer(variant_phrase.exact_string, text["text"]):
                if debug > 0:
                    print('search_exact_phrases_without_word_boundaries - match:', match)
                variant_phrase = phrase_model.variant_index[phrase_string]
                main_phrase_string = phrase_model.is_variant_of[phrase_string]
                main_phrase = phrase_model.phrase_index[main_phrase_string]
                match = PhraseMatch(main_phrase, variant_phrase, phrase_string, match.start(),
                                    text_id=text["id"], ignorecase=ignorecase)
                yield add_exact_match_score(match)


def search_exact(phrase: Phrase, text: Dict[str, str], ignorecase: bool = False, use_word_boundaries: bool = True):
    search_string = phrase.extact_word_boundary_string if use_word_boundaries else phrase.exact_string
    if ignorecase:
        return re.finditer(search_string, text["text"], flags=re.IGNORECASE)
    else:
        return re.finditer(search_string, text["text"])


def get_known_word_offsets(match_ranges: List[Dict[str, any]], text_doc: Dict[str, str]) -> Dict[int, dict]:
    known_word_offset = {}
    offset = match_ranges[0]["s"]
    for match_range in match_ranges:
        print(match_range)
        print("offset:", offset)
        match_text = text_doc["text"][match_range["s"]:match_range["e"]]
        match_words = re.split(r"\W+", match_text)
        for match_word in match_words:
            start = offset + text_doc["text"][offset:].index(match_word)
            end = start + len(match_word)
            known_word = {
                "match_word": match_word,
                "start": start,
                "end": end,
                "match_phrases": set([phrase for phrase in match_range["phrases"]])
            }
            match_word_offset = offset + text_doc["text"][offset:].index(match_word)
            offset = match_word_offset + len(match_word)
            print(known_word)
            known_word_offset[known_word["start"]] = known_word
    return known_word_offset


def get_exact_match_ranges(exact_matches: List[PhraseMatch]) -> List[dict]:
    sorted_matches = sorted(exact_matches, key=lambda m: m.offset)
    match_ranges = []
    if len(exact_matches) == 0:
        return []
    first_match = exact_matches[0]
    match_range = {"s": first_match.offset, "e": first_match.end, "phrases": {first_match.phrase.phrase_string}}
    for mi, match in enumerate(sorted_matches[1:]):
        if match.offset > match_range["e"]:
            match_ranges.append(match_range)
            match_range = {"s": match.offset, "e": match.end, "phrases": set()}
        match_range["e"] = max(match_range["e"], match.end)
        match_range["phrases"].add(match.phrase.phrase_string)
        if match.phrase.phrase_string != match.variant.phrase_string:
            match_range["phrases"].add(match.variant.phrase_string)
    match_ranges.append(match_range)
    return match_ranges
