from typing import Dict, List, Union

import fuzzy_search
from fuzzy_search.match.phrase_match import PhraseMatch
from fuzzy_search.phrase.phrase_model import PhraseModel
from fuzzy_search.pattern.fuzzy_template import FuzzyTemplate, FuzzyTemplateElement, FuzzyTemplateGroupElement
from fuzzy_search.search.context_searcher import FuzzyContextSearcher


DEBUG = False


def share_label(object1: Union[PhraseMatch, FuzzyTemplateElement],
                object2: Union[PhraseMatch, FuzzyTemplateElement]) -> bool:
    """Check if two fuzzy objects (phrase matches of template elements) share at least one label.

    :param object1: the first object to compare
    :type object1: Union[PhraseMatch, FuzzyTemplateElement]
    :param object2: the second object to compare
    :type object2: Union[PhraseMatch, FuzzyTemplateElement]
    :return: boolean value indicating that the two objects share a label
    :rtype: bool
    """
    label1 = set(object1.label) if isinstance(object1.label, list) else {object1.label}
    label2 = set(object2.label) if isinstance(object2.label, list) else {object2.label}
    # print("set 1:", label1)
    # print("set 2:", label2)
    return len(label1.intersection(label2)) > 0


def get_phrase_match_list_labels(phrase_matches: List[PhraseMatch]) -> List[str]:
    """Return a list of all the labels of a list of phrase matches.

    :param phrase_matches: a list of phrase matches
    :type phrase_matches: List[PhraseMatch]
    :return: a list of phrase match labels
    :rtype: List[str]
    """
    labels: List[str] = []
    for phrase_match in phrase_matches:
        labels += phrase_match.label if isinstance(phrase_match.label, list) else [phrase_match.label]
    return labels


def has_required_matches(phrase_matches: List[PhraseMatch], template: FuzzyTemplate) -> bool:
    """Check if list of phrase matches contain all required labels of a template.

    :param phrase_matches: a list of phrase matches
    :type phrase_matches: List[PhraseMatch]
    :param template: a fuzzy template to use for searching
    :type template: FuzzyTemplate
    :return: a True value only if all required labels have at least one match
    """
    phrase_match_labels = get_phrase_match_list_labels(phrase_matches)
    for required_label in template.get_required_labels():
        if required_label not in phrase_match_labels:
            return False
    return True


class TemplateMatch:

    def __init__(self, template: FuzzyTemplate, phrase_matches: List[PhraseMatch],
                 template_sequence: Dict[str, any]):
        """A match object for a given template, with a list of phrase matches that fill the template elements.

        :param template: the template that is matched
        :type template: FuzzyTemplate
        :param phrase_matches: the phrase matches that correspond to the template elements
        :type phrase_matches: List[PhraseMatch]
        :param template_sequence: a template sequence mapping each phrase match to the corresponding template labels
        :type template_sequence: Dict[str, any]
        """
        self.template = template
        self.phrase_matches = phrase_matches
        self.template_sequence = template_sequence
        self.element_matches = get_sequence_label_element_matches(template_sequence)

    def __repr__(self):
        return f"{self.__class__.__name__}(template={self.template}, element_matches={self.element_matches})"


def find_next_element_start_index(phrase_matches: List[PhraseMatch],
                                  template_element: FuzzyTemplateElement,
                                  template_start_index: int) -> int:
    """Find the next phrase match that matches a template element, from a given starting point
    in a list of phrase matches.

    :param phrase_matches: a list of phrase matches to be tested against a template element
    :type phrase_matches: List[PhraseMatch]
    :param template_element: a template element to test the phrase matches against
    :type template_element: FuzzyTemplateElement
    :param template_start_index: the index in the phrase list to start the matching process
    :type template_start_index: int
    :return: the index in the phrase list where the template element matches
    :rtype: int
    """
    # print("finding start for template element:", template_element.label, template_element.type, template_start_index)
    # if isinstance(template_element, FuzzyTemplateGroupElement):
        # print(template_element.group_element_labels)
    # print("finding start index:")
    for phrase_match in phrase_matches[template_start_index:]:
        if share_label(template_element, phrase_match):
            # print(phrase_matches.index(phrase_match), "phrase_match label:", phrase_match.label)
            # print(phrase_matches.index(phrase_match), "shared label")
            # print("returning start for template element:", template_element.label, template_element.type,
            #       template_start_index)
            return phrase_matches.index(phrase_match)
    return -1


def find_next_element_end_index(phrase_matches: List[PhraseMatch],
                                template_element: FuzzyTemplateElement,
                                element_start_index: int) -> int:
    """Find the next phrase match that doesn't match a template element, from a given starting point
    in a list of phrase matches.

    :param phrase_matches: a list of phrase matches to be tested against a template element
    :type phrase_matches: List[PhraseMatch]
    :param template_element: a template element to test the phrase matches against
    :type template_element: FuzzyTemplateElement
    :param element_start_index: the index in the phrase list where the template elements first matches the template
    :type element_start_index: int
    :return: the index in the phrase list where the template element stops matching
    :rtype: int
    """
    if element_start_index == -1:
        return -1
    # print("template element finding end:", template_element.label, template_element.type)
    for phrase_match in phrase_matches[element_start_index:]:
        if not share_label(template_element, phrase_match):
            # this is the first phrase match after the start index that does not fit
            # the template element, so the sequence ends here
            # print("element end index:", phrase_matches.index(phrase_match))
            return phrase_matches.index(phrase_match)
    # if all phrases matches after element start fit the element's label(s),
    # the end index is the end of the phrase matches
    return len(phrase_matches)


def initialize_sequence(element: FuzzyTemplateElement, start_index: int, end_index: int) -> Dict[str, any]:
    return {
        "element_label": element.label,
        "element_type": element.type,
        "element": element,
        "start": start_index,
        "end": end_index,
        "phrase_matches": [],
        "contains_required": False,
        "element_sequences": []
    }


def get_sequence_label_element_matches(template_sequence: Dict[str, any]) -> List[Dict[str, any]]:
    if template_sequence["element_type"] == "label":
        # print("label:", template_sequence["element_label"], template_sequence["start"], template_sequence["end"])
        return [{"label": template_sequence["element_label"], "phrase_matches": template_sequence["phrase_matches"]}]
    else:
        # print("group:", template_sequence["element_label"], template_sequence["start"], template_sequence["end"])
        template_phrase_matches: List[Dict[str, any]] = []
        for element_sequence in template_sequence["element_sequences"]:
            element_phrase_matches = get_sequence_label_element_matches(element_sequence)
            for element_phrase_match in element_phrase_matches:
                template_phrase_matches.append(element_phrase_match)
                if template_sequence["element_label"]:
                    if "label_groups" not in element_phrase_match:
                        element_phrase_match["label_groups"] = []
                    element_phrase_match["label_groups"].append(template_sequence["element_label"])
                # print("\telement:", element_phrase_match)
        return template_phrase_matches


def find_next_ordered_group_match_sequence(phrase_matches: List[PhraseMatch],
                                           template_group: FuzzyTemplateGroupElement,
                                           template_start_index: int) -> Union[None, Dict[str, any]]:
    """Find the next sequence of phrase matches that match an ordered template group element, from a given
    starting point in the list of phrase matches. This function returns None if the template doesn't match.

    :param phrase_matches: a list of phrase matches to be tested against a template element
    :type phrase_matches: List[PhraseMatch]
    :param template_group: a template group element to test the phrase matches against
    :type template_group: FuzzyTemplateGroupElement
    :param template_start_index: the index in the phrase list to start the matching process
    :type template_start_index: int
    :return: a sequence with start and end indexes in the list of phrase matches that match the template group
    :rtype: Union[None, Dict[str, any]]
    """
    # find first phrase match that belongs to first element of group
    group_sequence = initialize_sequence(template_group, template_start_index, template_start_index)
    element_sequences = []
    if DEBUG:
        print("ordered template group label:", template_group.label, "start index:", template_start_index)
    for group_element in template_group.elements:
        if DEBUG:
            print("ordered group element label:", group_element.label)
        if isinstance(group_element, FuzzyTemplateGroupElement):
            element_sequence = find_next_group_match_sequence(phrase_matches, group_element, group_sequence["end"])
            if DEBUG:
                print("returned sequence within ordered template:", group_element.label, element_sequence)
                if element_sequence:
                    group_sequence["element_sequences"].append(element_sequence)
        else:
            element_start_index = find_next_element_start_index(phrase_matches,
                                                                group_element, group_sequence["start"])
            element_end_index = find_next_element_end_index(phrase_matches,
                                                            group_element, element_start_index)
            if DEBUG:
                print(group_element.label, "element_start_index:", element_start_index,
                      "element_end_index:", element_end_index)
            element_sequence = initialize_sequence(group_element, element_start_index, element_end_index)
            element_sequence["phrase_matches"] = phrase_matches[element_start_index:element_end_index]
        element_sequences.append(element_sequence)
        if element_sequence:
            if DEBUG:
                print(f"ordered group element {group_element.label} indexes:", element_sequence["start"], element_sequence["end"])
                print("\tgroup start,end:", group_sequence["start"], group_sequence["end"])
        if (not element_sequence or element_sequence["start"] == -1) and not group_element.required:
            if DEBUG:
                print(template_group.label, "\tnot found, not required")
            # this element has no matches in the remaining match phrases but is not required so can be skipped
            continue
        elif (not element_sequence or element_sequence["start"] == -1) and group_element.required:
            if DEBUG:
                print(template_group.label, "\tnot found, but required, returning None", group_element.label)
            # this element is required but has no matches, so the phrase matches do not fit the group template
            return None
        elif group_sequence["start"] == group_sequence["end"]:
            # The group is still empty and this element has matches, so set its start and end
            # as the groups start and end sequence
            group_sequence["element_sequences"].append(element_sequence)
            if DEBUG:
                print('adding element sequence:', element_sequence)
                print(template_group.label, "\tfirst found", group_element.label, element_sequence["start"],
                      element_sequence["end"])
                print(template_group.label, "\tcurrent group sequence:", group_sequence["start"], group_sequence["end"])
            group_sequence["start"] = element_sequence["start"]
            group_sequence["end"] = element_sequence["end"]
            if DEBUG:
                print(template_group.label, "\tupdated group sequence:", group_sequence["start"], group_sequence["end"])
            if group_element.required:
                group_sequence["contains_required"] = True
        elif element_sequence["start"] <= group_sequence["end"]:
            # there is no gap between this element's phrase matches and
            # those of the previous elements in the group so update end
            # if this element's matches extend the group end
            group_sequence["element_sequences"].append(element_sequence)
            if DEBUG:
                print('adding element sequence:', element_sequence)
                print(template_group.label, "\tnext found", group_element.label, element_sequence["start"],
                      element_sequence["end"])
                print(template_group.label, "\tcurrent group sequence:", group_sequence["start"], group_sequence["end"])
            if element_sequence["end"] > group_sequence["end"]:
                group_sequence["end"] = element_sequence["end"]
            if DEBUG:
                print(template_group.label, "\tnext sequence:", group_sequence["start"], group_sequence["end"])
        elif group_element.required and element_sequence["start"] > group_sequence["end"]:
            # there is a gap between this element's matches and the previous sequence, but this element is
            # required. Check if parts of the previous sequence are required. If so, the phrase matches do
            # not fit the template group. If not, the previous sequence can be ignored and this required
            # element is the start of the sequence
            if group_sequence["contains_required"]:
                if DEBUG:
                    print(template_group.label, "next required disconnected from previous required")
                return None
            else:
                if DEBUG:
                    print('replacing previous element sequences with current element sequence:', element_sequence)
                group_sequence["element_sequences"] = [element_sequence]
                group_sequence["start"] = element_sequence["start"]
                group_sequence["end"] = element_sequence["end"]
                group_sequence["contains_required"] = True
                if DEBUG:
                    print(template_group.label, "\treset first found", group_element.label, element_sequence["start"],
                          element_sequence["end"])
                    print(template_group.label, "\treset first sequence:", group_sequence["start"], group_sequence["end"])
        else:
            # there are phrase matches in between this element and the previous elements,
            # and this element is not required, so skip this element
            if DEBUG:
                print(template_group.label, "gap between this non-required element and previous sequence. " +
                      "Skipping this element:", group_element.label, element_sequence["start"], element_sequence["end"])
                print('remove element sequence:', element_sequence)
                group_sequence['element_sequences'].pop(-1)
                print('group sequence:', group_sequence['element_sequences'])
            pass
            # return None
    if group_sequence["start"] == -1:
        # none of the group's elements has a phrase match sequence, so the template has no matches
        if DEBUG:
            print(template_group.label, "returning no ordered sequence")
        return None
    if DEBUG:
        print(template_group.label, "returning ordered sequence:", group_sequence["start"], group_sequence["end"])
    return group_sequence


def find_next_unordered_group_match_sequence(phrase_matches: List[PhraseMatch],
                                             template_group: FuzzyTemplateGroupElement,
                                             template_start_index: int) -> Union[None, Dict[str, any]]:
    """Find the next sequence of phrase matches that match an unordered template group element, from a given
    starting point in the list of phrase matches. This function returns None if the template doesn't match.

    :param phrase_matches: a list of phrase matches to be tested against a template element
    :type phrase_matches: List[PhraseMatch]
    :param template_group: a template group element to test the phrase matches against
    :type template_group: FuzzyTemplateGroupElement
    :param template_start_index: the index in the phrase list to start the matching process
    :type template_start_index: int
    :return: a sequence with start and end indexes in the list of phrase matches that match the template group
    :rtype: Union[None, Dict[str, any]]
    """
    # find first phrase match that belongs to first element of group
    group_sequence = initialize_sequence(template_group, -1, -1)
    element_sequences = []
    # print("unordered template group label:", template_group.label, "start_index:", template_start_index)
    for group_element in template_group.elements:
        # print("unordered group element label:", group_element.label)
        if isinstance(group_element, FuzzyTemplateGroupElement):
            # print("checking group element next sequence")
            element_sequence = find_next_group_match_sequence(phrase_matches, group_element, template_start_index)
            # print("returned sequence within unordered template:", group_element.label, element_sequence)
            # if element_sequence:
            #     group_sequence["element_sequences"].append(element_sequence)
        else:
            # print("checking label element next sequence")
            element_start_index = find_next_element_start_index(phrase_matches,
                                                                group_element, template_start_index)
            element_end_index = find_next_element_end_index(phrase_matches,
                                                            group_element, element_start_index)
            element_sequence = initialize_sequence(group_element, element_start_index, element_end_index)
            # print("returned label sequence within unordered template:", group_element.label, element_start_index, element_end_index)
            element_sequence["phrase_matches"] = phrase_matches[element_start_index:element_end_index]
        # group_sequence["element_sequences"].append(element_sequence)
        # element_sequences.append(element_sequence)
        # print(f"unordered group element {group_element.label} indexes:", element_start_index, element_end_index)
        if element_sequence["start"] == -1 and not group_element.required:
            # print("\tnot found, not required")
            # this element has no matches, but is not required
            # so does not contribute to the current template sequence
            continue
        elif element_sequence["start"] == -1 and group_element.required:
            # print("\tnot found, but required")
            # this element has no matches and is required, so the phrase matches
            # do not fit this template group
            return None
        else:
            # this element has matches, so add its sequence to the list of sequences
            # print("\tunordered element sequence", group_element.label, element_start_index, element_end_index)
            # print("\tunordered group sequence:", group_sequence["start"], group_sequence["end"])
            element_sequences.append(element_sequence)
    # print(template_group.label, "sorting element sequences:", len(element_sequences))
    element_sequences.sort(key=lambda x: x["start"])
    for element_sequence in element_sequences:
        # print(template_group.label, "checking sorted element sequence", element_sequence["element_label"],
        #      element_sequence["start"], element_sequence["end"])
        if group_sequence["start"] == -1:
            # this is the first element with matches, so set its sequence as the group sequence
            # print("adding first sequence")
            group_sequence["start"] = element_sequence["start"]
            group_sequence["end"] = element_sequence["end"]
            group_sequence["element_sequences"].append(element_sequence)
        elif element_sequence["start"] <= group_sequence["end"] < element_sequence["end"]:
            # there are no matches in between this element sequence and the previous one,
            # so adjust the end of the group sequence to contain include this element sequence as well
            # print("extending end")
            group_sequence["end"] = element_sequence["end"]
            group_sequence["element_sequences"].append(element_sequence)
        elif element_sequence["end"] <= group_sequence["end"]:
            # this sequences falls within the current group sequence, so add it but leave the group sequence end
            # print("not extending end")
            group_sequence["element_sequences"].append(element_sequence)
        elif element_sequence["start"] > group_sequence["end"] and element_sequence["element"].required:
            # there is a gap between this element's matches and the previous sequence, but this element is
            # required. Check if parts of the previous sequence are required. If so, the phrase matches do
            # not fit the template group. If not, the previous sequence can be ignored and this required
            # element is the start of the sequence
            if group_sequence["contains_required"]:
                # print("next required disconnected from previous required")
                return None
            # the previous sequence has no required element, and this element is required
            # so the matches of the previous elements do not fit this template group
            group_sequence["start"] = element_sequence["start"]
            group_sequence["end"] = element_sequence["end"]
            group_sequence["element_sequences"] = [element_sequence]
        elif element_sequence["start"] > group_sequence["end"]:
            # there are matches in between this element and the previous so this element's
            # matches are part of a next sequence
            break
        else:
            print(template_group.label, "unknown condition")
            print(template_group.label, "group start,end:", group_sequence["start"], group_sequence["end"])
            print(template_group.label, "element start,end:", element_sequence["start"], element_sequence["end"])
            raise ValueError("UNKNOWN CONDITION")
    if group_sequence["start"] == -1:
        # none of the group's elements has a phrase match sequence, so the template has no matches
        # print("returning no unordered sequence:", template_group.label)
        return None
    # print(template_group.label, "returning unordered sequence:", group_sequence["start"], group_sequence["end"])
    return group_sequence


def find_next_group_match_sequence(phrase_matches: List[PhraseMatch],
                                   template_group: FuzzyTemplateGroupElement,
                                   template_start_index: int) -> Union[None, Dict[str, any]]:
    """Find the next sequence of phrase matches that match a template group element, from a given
    starting point in the list of phrase matches. This function returns None if the template doesn't match.

    :param phrase_matches: a list of phrase matches to be tested against a template element
    :type phrase_matches: List[PhraseMatch]
    :param template_group: a template group element to test the phrase matches against
    :type template_group: FuzzyTemplateGroupElement
    :param template_start_index: the index in the phrase list to start the matching process
    :type template_start_index: int
    :return: a sequence with start and end indexes in the list of phrase matches that match the template group
    :rtype: Union[None, Dict[str, any]]
    """
    if template_group.ordered:
        if DEBUG:
            print(template_group.label, "CHECKING FOR FIRST ORDERED SEQUENCE")
        sequence = find_next_ordered_group_match_sequence(phrase_matches, template_group, template_start_index)
        if DEBUG:
            print("returned ordered sequence")
            if sequence:
                print(template_group.label, "\treceived ordered start,end:", sequence["start"], sequence["end"])
            else:
                print(template_group.label, "\treceived no ordered sequence")
        return sequence
    else:
        if DEBUG:
            print(template_group.label, "CHECKING FOR FIRST UNORDERED SEQUENCE")
        sequence = find_next_unordered_group_match_sequence(phrase_matches, template_group, template_start_index)
        if DEBUG:
            if sequence:
                print(template_group.label, "\treceived first start,end:", sequence["start"], sequence["end"])
                print(template_group.label, "CHECKING FOR ADDITIONAL UNORDERED SEQUENCE")
            else:
                print(template_group.label, "no first sequence for")
        while True and sequence:
            next_sequence = find_next_unordered_group_match_sequence(phrase_matches, template_group, sequence["end"])
            if not next_sequence:
                if DEBUG:
                    print(template_group.label, "\treceived no next sequence")
                break
            if next_sequence["start"] != sequence["end"]:
                if DEBUG:
                    print(template_group.label, "\treceived sequence with gap next start,end:", next_sequence["start"], next_sequence["end"])
                break
            elif next_sequence["end"] > sequence["end"]:
                if DEBUG:
                    print(template_group.label, "\treceived sequence with no gap next start,end:", next_sequence["start"], next_sequence["end"])
                sequence["end"] = next_sequence["end"]
                sequence["element_sequences"] += next_sequence["element_sequences"]
            else:
                if DEBUG:
                    print(template_group.label, "\treceived something strange:", next_sequence["start"], next_sequence["end"])
                break
        if DEBUG:
            if sequence:
                print(template_group.label, "RETURN find_next_group_match_sequence:", sequence["start"], sequence["end"])
            else:
                print(template_group.label, "RETURN find_next_group_match_sequence:", sequence)
        return sequence


class FuzzyTemplateSearcher(FuzzyContextSearcher):

    def __init__(self, template: Union[None, FuzzyTemplate] = None, config: Union[None, dict] = None):
        """A fuzzy searcher for finding fuzzy matches in texts and checking if the matches fit a given template.
        The FuzzyTemplateSearcher incorporates a FuzzyContextSearcher for searching phrase matches in texts.
        The phrases are taken from the phrase model that is part of the template. The FuzzyContextSearcher
        uses the default configuration unless a searcher_config is specified that overrides specific properties.

        :param template: a fuzzy template to use for searching
        :type template: FuzzyTemplate
        :param searcher_config: an optional configuration dictionary to configure the FuzzyTemplateSearcher
        """
        super().__init__(config=config)
        self.__version__ = fuzzy_search.__version__
        self.template: Union[None, FuzzyTemplate] = template if template else None
        self.phrase_model: Union[None, PhraseModel] = template.phrase_model if template else None
        if self.phrase_model:
            self.index_phrase_model(self.phrase_model)

    def __repr__(self):
        return f"{self.__class__.__name__}(template={self.template})"

    def set_template(self, template: FuzzyTemplate) -> None:
        """Set a new template for the searcher and index the corresponding phrase model.

        :param template: a fuzzy template to use for searching
        :type template: FuzzyTemplate
        """
        self.template = template
        self.phrase_model = template.phrase_model
        self.index_phrase_model(self.phrase_model)

    def search_text(self, text: Union[str, Dict[str, str]]) -> List[TemplateMatch]:
        """Search phrases from the registered template's phrase model in the text and check if the resulting matches
        together match the template. This method returns a dictionary including the individual phrase matches and
        any template matches.

        :param text: a text to search in, either as a string or a dictionary with text and an identifier
        :type text: Union[str, Dict[str, str]]
        :return: a dictionary with all phrase matches and template matches
        :rtype: Dict[str, Union[List[PhraseMatch], List[TemplateMatch]]]
        """
        if not self.template:
            raise ValueError("No fuzzy search template registered.")
        phrase_matches = self.find_matches(text)
        template_matches = self.find_template_matches(phrase_matches)
        return template_matches
        # return {"phrase_matches": phrase_matches, "template_matches": template_matches}

    def filter_phrase_matches(self, phrase_matches: List[PhraseMatch]) -> List[PhraseMatch]:
        """Filter a list of phrase matches to only include phrase matches that have at least one label in
        common with the template.

        :param phrase_matches: a list of phrase matches
        :type phrase_matches: List[PhraseMatch]
        :return: a filtered list of phrases matches
        :rtype: List[PhraseMatch]
        """
        # first, check if required elements are present
        if not has_required_matches(phrase_matches, self.template):
            return []
        # second, remove phrase matches that have no label that is part of the template
        return [phrase_match for phrase_match in phrase_matches if self.template.has_label(phrase_match.label)]

    def find_template_matches(self, phrase_matches: List[PhraseMatch]) -> List[TemplateMatch]:
        """Find all the matches that fit a template. The method returns a list of template matches, where
        each template match contains the phrase match that fit the template. There can be multiple
        template matches, if the phrase matches fit a template multiple times.

        :param phrase_matches: a list of phrase matches
        :type phrase_matches: List[PhraseMatch]
        :return: a list of template matches
        :rtype: List[TemplateMatch]
        """
        template_matches: List[TemplateMatch] = []
        # make sure the matches are sorted in order of occurrence in the text
        template_phrase_matches = self.filter_phrase_matches(sorted(phrase_matches, key=lambda x: x.offset))
        sequence_start_index = 0
        # print("num matches:", len(template_phrase_matches))
        # for phrase_match in template_phrase_matches:
        #     print("\t", phrase_match.label, phrase_match.phrase.phrase_string, "\t", phrase_match.string)
        while sequence_start_index < len(template_phrase_matches):
            if DEBUG:
                print("sequence_start_index:", sequence_start_index)
            template_sequence = find_next_group_match_sequence(template_phrase_matches, self.template.root_element,
                                                               sequence_start_index)
            if template_sequence is None:
                break
            else:
                sequence_start_index = template_sequence["end"]
                sequence_matches = phrase_matches[template_sequence["start"]:template_sequence["end"]]
                template_match = TemplateMatch(template=self.template, phrase_matches=sequence_matches,
                                               template_sequence=template_sequence)
                template_matches.append(template_match)
            if DEBUG:
                print("template_sequence:", template_sequence["start"], template_sequence["end"])
            # print("updated sequence_start_index:", sequence_start_index)
            # print("\n\n")
        return template_matches
