from typing import Union

dutch_person_name_patterns = {
    "name_suffixes": r"( de jonge| de oude| junior| senior)",
    "name_pattern": r"(([A-Z](\w|-)+)( [A-Z](\w|-)+)*( van| de| der| den)*( [A-Z](\w|-)+)+( de jonge| de oude| junior| senior)?)",
    "name_comma_name_pattern": r"((([A-Z](\w|-)+)( [A-Z](\w|-)+)*( van| de| der| den)*( [A-Z](\w|-)+)+( de jonge| de oude| junior| senior)?), ?)+",
    "name_comma_pattern": r"(([A-Z](\w|-)+)( [A-Z](\w|-)+)*( van| de| der| den)*( [A-Z](\w|-)+)+( de jonge| de oude| junior| senior)?), ?",
}

dutch_date_patterns = {
    "week_day": r"(maandag|dinsdag|woensdag|donderdag|vrijdag|zaterdag|zondag)",
    "month": r"(jan(uari|.)?|feb(ruari|\.)?|maart|mrt|apr(il|\.)|mei|jun(i)?|jul(i)?|aug(ustus|\.)?|sep(t|t\.|tember|\.)?|okt(ober|\.)?|nov(ember|\.)?|dec(ember|\.)?)",
    "day_month": r"(\d{1,2}) (jan(uari|.)?|feb(ruari|\.)?|maart|mrt|apr(il|\.)|mei|jun(i)?|jul(i)?|aug(ustus|\.)?|sep(t|t\.|tember|\.)?|okt(ober|\.)?|nov(ember|\.)?|dec(ember|\.)?)",
    "year": r"(\d{4})",
    "time": r"\b(\d{1,2}|een|twee|drie|vier|vijf|zes|zeven|acht|negen|tien|elf|twaalf) (uu?ren)\b",
    "day_part": r"\b's (avonds|middags|ochtends)",
}

dutch_place_name_patterns = {
    "name_pattern": r"([A-Z](\w|-)+)",
}

dutch_place_name_patterns["in_placename"] = r"(in|tot) " + dutch_place_name_patterns["name_pattern"]

dutch_date_patterns["weekday_comma_day_month"] = dutch_date_patterns["week_day"] + "(,? (de|den)?) " + \
                                                 dutch_date_patterns["day_month"]

dutch_person_name_patterns["name_and_name_pattern"] = dutch_person_name_patterns["name_pattern"] + " en " + \
                                                      dutch_person_name_patterns["name_pattern"]
dutch_person_name_patterns["name_sequence_pattern"] = dutch_person_name_patterns["name_comma_pattern"] + \
                                                      dutch_person_name_patterns["name_and_name_pattern"]

# These pattern definitions specify:
# 1. which regex patterns to use for which pattern names
# 2. which regex match groups should be extracted
pattern_definitions = {
    "name": {
        "pattern": dutch_person_name_patterns["name_pattern"],
        "group_indices": [1],
        "type": "dutch_person_name",
    },
    "name_and_name": {
        "pattern": dutch_person_name_patterns["name_and_name_pattern"],
        "group_indices": [1, 10],
        "type": "dutch_person_name",
    },
    "name_sequence": {
        "pattern": dutch_person_name_patterns["name_sequence_pattern"],
        "group_indices": [1, 10, 19],
        "type": "dutch_person_name",
    },
    # commented out because it's not useful on it own:
    # "name_comma_name": {
    #    "pattern": dutch_person_name_patterns["name_comma_name_pattern"],
    #    "group_indices": [1, 10, 19],
    #    "type": "dutch_person_name",
    # },
    "weekday_comma_day_month": {
        "pattern": dutch_date_patterns["weekday_comma_day_month"],
        "group_indices": [1, 4, 5],
        "type": "dutch_date",
    },
}


def list_context_pattern_types(context_type=None):
    if not context_type:
        context_type = "all"
    if context_type not in context_pattern:
        print("ERROR - Unknown context type. Pick from:")
        for context_type in context_pattern:
            print("\t{t}".format(context_type))
        raise KeyError("Unknown context type")
    return [pattern_type for pattern_type in context_pattern[context_type]]


def list_pattern_names(name_only=True, pattern_type=None):
    if pattern_type:
        return [pattern_name for pattern_name in pattern_definitions if
                pattern_definitions[pattern_name]["type"] == pattern_type]
    else:
        return [pattern_name for pattern_name in pattern_definitions]


def list_pattern_definitions(pattern_type=None):
    if pattern_type:
        return [pattern_definitions[pattern_name] for pattern_name in pattern_definitions if
                pattern_definitions[pattern_name]["type"] == pattern_type]
    else:
        return pattern_definitions


def pattern_comma_then_context(name, pattern_definition, context_string):
    return {
        "name": name + "_comma_then_context",
        "pattern": pattern_definition["pattern"] + " ?, ?" + context_string,
        "group_indices": pattern_definition["group_indices"],
    }


def context_then_pattern(name, pattern_definition, context_string):
    return {
        "name": "context_then_" + name,
        "pattern": context_string + ",? " + pattern_definition["pattern"],
        "group_indices": pattern_definition["group_indices"],
    }


def pattern_before_context(name, pattern_definition, context_string, max_distance=10):
    return {
        "name": name + "_before_context",
        "pattern": pattern_definition["pattern"] + ".{d}".format(d=max_distance) + context_string,
        "group_indices": pattern_definition["group_indices"],
    }


def context_before_pattern(name, pattern_definition, context_string, max_distance=10):
    return {
        "name": "context_before_" + name,
        "pattern": context_string + ".{d}".format(d=max_distance) + pattern_definition["pattern"],
        "group_indices": pattern_definition["group_indices"],
    }


def get_search_patterns(pattern_type=None):
    if pattern_type:
        return {pattern_name: pattern_definitions[pattern_name] for pattern_name in pattern_definitions if
                pattern_definitions[pattern_name]["type"] == pattern_type}
    else:
        return pattern_definitions


def get_context_patterns(context_type: Union[None, str] = None) -> dict:
    if not context_type:
        context_type = "all"
    if context_type not in context_pattern:
        print("ERROR - Unknown context type. Pick from:")
        for context_type in context_pattern:
            print(f"\t{context_type}")
        raise KeyError("Unknown context type")
    return context_pattern[context_type]


# dictionary mapping regex pattern name to function to create context-specific regex pattern
context_pattern = {
    "person_name": {
        "pattern_comma_then_context": pattern_comma_then_context,
        "context_then_pattern": context_then_pattern,
    },
    "distance": {
        "pattern_before_context": pattern_before_context,
        "context_before_pattern": context_before_pattern,
    },
    "all": {
        "pattern_comma_then_context": pattern_comma_then_context,
        "context_then_pattern": context_then_pattern,
        "pattern_before_context": pattern_before_context,
        "context_before_pattern": context_before_pattern,
    },
}


def escape_string(string):
    string = string.replace("\\", r"\\").replace("/", r"\/")
    string = string.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(").replace(")", r"\)")
    string = string.replace("{", r"\{").replace("}", r"\}")
    string = string.replace("*", r"\*").replace("?", r"\?").replace("+", r"\+")
    string = string.replace(".", r"\.").replace("|", r"\|")
    string = string.replace("!", r"\!").replace("^", r"\^").replace("$", r"\$")
    return string


def make_search_context_patterns(context_string, pattern_names, context_patterns):
    context_string = escape_string(context_string)
    patterns = []
    for context_pattern in context_patterns:
        for pattern_name in pattern_names:
            patterns += [
                context_patterns[context_pattern](pattern_name, pattern_definitions[pattern_name], context_string)]
    return patterns


if __name__ == "__main__":
    pattern_names = ["name_comma_name"]
    pattern_types = ["pattern_comma_then_context"]
    context_string = "Makelaar"

    make_search_context_patterns(context_string, pattern_names, pattern_types)
