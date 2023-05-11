from __future__ import annotations
from typing import Dict, List, Set, Union

from fuzzy_search.phrase.phrase_model import PhraseModel
from fuzzy_search.phrase.phrase import Phrase


def validate_element_properties(label: str, required: bool = False, cardinality: str = "multi",
                                next_label: Union[None, str, List[str]] = None,
                                next_distance_max: Union[None, int] = None,
                                variable: bool = False) -> None:
    """Validate the properties of a FuzzyTemplate element.

    :param label: the label of the element, which can be a single string or a list of strings
    :type label: Union[str, List[str]]
    :param required: whether or not the element must match for the template to match
    :type required: bool
    :param cardinality: whether the element can occur only once (default) or multiple times in a template match.
    :type cardinality: str
    :param next_label: what the label of the next element should be. Use a list of labels for multiple options.
    :type next_label: Union[str, List[str]]
    :param next_distance_max: the maximum distance allowed between this element and the next element in the template
    :type next_distance_max: int
    :param variable: flag to indicate the element has no phrases but has variable text (default is False)
    :type variable: bool
    """
    if not isinstance(label, str):
        raise ValueError("label must be string")
    if required is None:
        pass
    elif not isinstance(required, bool):
        raise ValueError("'required' property must be a boolean value")
    if cardinality is None:
        pass
    elif not isinstance(cardinality, str) or cardinality not in ["single", "multi"]:
        raise ValueError("cardinality must be a string with either 'single' or 'multi' as value")
    if next_label is None:
        pass
    elif isinstance(next_label, list):
        for label in next_label:
            if not isinstance(label, str):
                raise ValueError("next_label list items must be strings")
    elif not isinstance(next_label, str):
        raise ValueError("next_label must be string or list of strings")
    if next_distance_max is None:
        pass
    elif not isinstance(next_distance_max, int) or next_distance_max < 0:
        raise ValueError("next_distance_max must be an positive integer")
    if not isinstance(variable, bool):
        raise ValueError("'variable' flag must be a boolean")


class FuzzyTemplateElement:

    def __init__(self, label: Union[None, str, List[str]], element_type: str, required: bool):
        self.label = label
        self.type = element_type
        self.required = required


class FuzzyTemplateLabelElement(FuzzyTemplateElement):

    def __init__(self, label: str, required: bool = False, cardinality: str = "single",
                 next_label: Union[None, str, List[str]] = None, next_distance_max: Union[None, int] = None,
                 variable: bool = False):
        """A FuzzyTemplate element with properties to define its role in the template it is part of.
        The 'label' property is the only required property. All elements must have either a single string or
        list of strings as label.

        :param label: the label of the element, which can be a single string or a list of strings
        :type label: Union[str, List[str]]
        :param required: whether or not the element must match for the template to match
        :type required: bool
        :param cardinality: whether the element can occur only once or multiple times in a template match. The default
        value is 'multi'
        :type cardinality: str
        :param next_label: what the label of the next template element should be. This can be a list of labels to
        allow different types of element to come next
        :type next_label: Union[str, List[str]]
        :param next_distance_max: the maximum distance allowed between this element and the next element in the template
        :type next_distance_max: int
        :param variable: flag to indicate the element has no phrases but has variable text (default is False)
        :type variable: bool
        """
        validate_element_properties(label, required, cardinality, next_label, next_distance_max)
        super().__init__(label, "label", required)
        self.cardinality = cardinality if cardinality is not None else "single"
        self.next_label = next_label
        self.next_distance_max = next_distance_max
        self.variable = variable

    def __repr__(self):
        return f"FuzzyTemplateElement(label='{self.label}', required={self.required}, cardinality='{self.cardinality}'"


class FuzzyTemplateGroupElement(FuzzyTemplateElement):

    def __init__(self, elements: List[FuzzyTemplateElement],
                 label: Union[str, None] = None, ordered: bool = True, required: bool = False):
        super().__init__(label, "group", required)
        self.ordered = ordered
        self.elements = elements
        self.group_element_labels: Set[str] = set()
        self.has_variable_element = False
        for element in self.elements:
            if isinstance(element, FuzzyTemplateGroupElement):
                self.group_element_labels = self.group_element_labels.union(element.group_element_labels)
                self.has_variable_element = element.has_variable_element
            else:
                self.group_element_labels.add(element.label)
                if isinstance(element, FuzzyTemplateLabelElement) and element.variable:
                    self.has_variable_element = True
            if element.required:
                # override the received required value if one of its sub-elements is required
                self.required = True

    def __repr__(self):
        return f"FuzzyTemplateGroup(label='{self.label}', required={self.required}, ordered='{self.ordered}'"


def generate_label_from_json(label: str, element_info: dict) -> FuzzyTemplateLabelElement:
    """Generate a FuzzyTemplateLabelElement from a label and an element json dictionary.

    :param label: the label string for the label element
    :type label: str
    :param element_info: a dictionary containing the properties of the template label element
    :type element_info: dict
    :return: a fuzzy template label element
    :rtype: FuzzyTemplateLabelElement
    """
    required = element_info["required"] if "required" in element_info else False
    cardinality = element_info["cardinality"] if "cardinality" in element_info else "single"
    next_label = element_info["next_label"] if "next_label" in element_info else None
    next_distance_max = element_info["next_distance_max"] if "next_distance_max" in element_info else None
    variable = element_info["variable"] if "variable" in element_info else False
    return FuzzyTemplateLabelElement(label, required, cardinality, next_label, next_distance_max, variable)


def generate_group_from_json(element_info: dict,
                             group_elements: List[FuzzyTemplateElement]) -> FuzzyTemplateGroupElement:
    """Generate a FuzzyTemplateGroupElement from a element json dictionary and a list of group elements.

    :param element_info: a dictionary containing the properties of the template group element
    :type element_info: dict
    :param group_elements: a list of fuzzy template elements that are part of the group element
    :type group_elements: List[FuzzyTemplateElement]
    :return: a fuzzy template group element
    :rtype: FuzzyTemplateGroupElement
    """
    label = element_info["label"] if "label" in element_info else None
    ordered = element_info["ordered"] if "ordered" in element_info else True
    required = element_info["required"] if "required" in element_info else False
    return FuzzyTemplateGroupElement(group_elements, label=label, ordered=ordered, required=required)


class FuzzyTemplate:

    def __init__(self, phrase_model: PhraseModel,
                 template_json: Union[List[str], List[dict], Dict[str, Union[str, dict]]],
                 ignore_unknown: bool = False, ordered: bool = False):
        """A fuzzy search template to register phrases from a phrase model as elements of the template.
        The template can be used in combination with a fuzzy template searcher to find a given set
        of phrases (the template) within a certain range of text. The order, cardinality of elements
        and the distance between them can be specified. Elements can be grouped into ordered or unordered
        blocks of elements. Element groups can be hierarchical.

        :param phrase_model: a PhraseModel object with phrases that correspond to the element label
        in the template.
        :type phrase_model: PhraseModel
        :param template_json: a dictionary of template groups or elements to be registered as part of the template
        :type template_json: Union[List[str], List[dict], Dict[str, Union[str, dict]]]
        :param ignore_unknown: whether to ignore elements with labels that are not in the phrase model
        :type ignore_unknown: bool
        """
        self.phrase_model = phrase_model
        self.ordered = ordered
        self.labels: Set[str] = set()
        self.elements: Set[FuzzyTemplateLabelElement] = set()
        self.groups: Set[FuzzyTemplateGroupElement] = set()
        self.required: Set[FuzzyTemplateLabelElement] = set()
        self.label_element_index: Dict[str, FuzzyTemplateLabelElement] = {}
        self.group_element_index: Dict[str, FuzzyTemplateGroupElement] = {}
        self.root_element = None
        self.ignore_unknown = ignore_unknown
        self.template_json = template_json
        self.template_name = None
        if 'label' in template_json:
            self.template_name = template_json['label']
        self.register_template(template_json)

    def __repr__(self):
        return f"FuzzyTemplate(labels={self.labels}, required={self.get_required_labels()})"

    def get_label_phrases(self, label: str) -> List[Phrase]:
        """Return a list of phrases that have a given label.

        :param label: a phrase label for phrases in the registered phrase_model
        :type label: str
        :return: a list of phrases from the registered phrase model that have a given phrase
        :rtype: List[Phrase]
        """
        if label not in self.phrase_model.is_label_of:
            return []
        return [self.phrase_model.phrase_index[phrase_string] for phrase_string in self.phrase_model.is_label_of[label]]

    def has_label(self, label: Union[str, List[str]], ) -> bool:
        """Check if the template has label elements with a given label or list of label (any or all).

        :param label: a fuzzy element label
        :type label: Union[str, List[str]]
        :return: whether the label corresponds to any registered element(s)
        :rtype: bool
        """
        if isinstance(label, list):
            for label_item in label:
                if label_item in self.label_element_index:
                    return True
            return False
        else:
            return label in self.label_element_index

    def has_group(self, group: str) -> bool:
        """Check if the template has group elements with a given group name.

        :param group: a fuzzy element group
        :type group: str
        :return: whether the group corresponds to any registered element(s)
        :rtype: bool
        """
        return group in self.group_element_index

    def get_element(self, element_label: str) -> Union[None, FuzzyTemplateLabelElement, FuzzyTemplateGroupElement]:
        """Return the element corresponding to a given label.

        :param element_label: a fuzzy element label
        :type element_label: str
        :return: the element corresponding to the label or None if label is unknown
        :rtype: Union[FuzzyTemplateElement]
        """
        if self.has_label(element_label):
            return self.label_element_index[element_label]
        elif self.has_group(element_label):
            return self.group_element_index[element_label]
        else:
            return None

    def register_template(self, template_json: Union[List[str], List[dict], Dict[str, Union[str, dict]]]) -> None:
        """Register a list of elements as a fuzzy template. Each element contains a label
        that corresponds to at least one phrase in the registered phrase model.

        :param template_json: a dictionary of template groups or elements to be registered as part of the template
        :type template_json: Union[List[str], List[dict], Dict[str, Union[str, dict]]]
        """
        if isinstance(template_json, list):
            # if the template is a list, turn it into a dictionary first
            elements: List[Dict[str, any]] = []
            for ele in template_json:
                if isinstance(ele, str):
                    ele = {"label": ele, "type": "label"}
                elements.append(ele)
            template_json = {
                "type": "group",
                "elements": elements
            }
        if "type" not in template_json:
            template_json["type"] = "group" if "elements" in template_json else "label"
        if template_json["type"] == "label":
            # make sure the top level is always a group
            template_json = {"type": "group", "elements": [template_json]}
        elif template_json["type"] != "group":
            raise ValueError("element type must be 'label' or 'group'.")
        self.root_element = self.parse_group_element(template_json)

    def get_required_elements(self) -> List[FuzzyTemplateLabelElement]:
        """Return all required elements in the template.

        :return: the list of labels of required elements
        :rtype: List[FuzzyTemplateElement]
        """
        return [element for element in self.required]

    def get_required_labels(self) -> List[str]:
        """Return the labels of all required elements in the template.

        :return: the list of labels of required elements
        :rtype: List[str]
        """
        return [element.label for element in self.required]

    def get_elements_by_cardinality(self, cardinality: str = "single") -> List[FuzzyTemplateLabelElement]:
        """Return all template elements with a given cardinality.

        :param cardinality: a cardinality type ('single' or 'multi')
        :type cardinality: str
        :return: the list of labels of elements with a given cardinality
        :rtype: List[str]
        """
        return [element for element in self.elements if element.cardinality == cardinality]

    def get_labels_by_cardinality(self, cardinality: str = "single") -> List[str]:
        """Return the labels of all template elements with a given cardinality.

        :param cardinality: a cardinality type ('single' or 'multi')
        :type cardinality: str
        :return: the list of labels of elements with a given cardinality
        :rtype: List[str]
        """
        return [element.label for element in self.elements if element.cardinality == cardinality]

    def parse_label_element(self, label_info: Union[str, Dict[str, any]]) -> Union[FuzzyTemplateLabelElement, None]:
        """Parse a label element dictionary/JSON object into a fuzzy template label element.

        :param label_info: a dictionary containing the properties of the template label element
        :type label_info: dict
        :return: a fuzzy template label element, or None if the label is not used in the phrase model
        :rtype: FuzzyTemplateLabelElement
        """
        if isinstance(label_info, str):
            label_info = {"label": label_info}
        if "label" not in label_info:
            print("element json without label:", label_info)
        label = label_info["label"]
        if not self.phrase_model.is_label(label):
            if "variable" in label_info and label_info["variable"] is True:
                pass
            elif self.ignore_unknown:
                # print("skipping unknown phrase label", label)
                return None
            else:
                raise ValueError(f"label '{label}' does not correspond to any phrase in registered phrase model")
        element = generate_label_from_json(label, label_info)
        # register the label element as part of the template
        self.label_element_index[element.label] = element
        self.labels.add(element.label)
        if element.required:
            # keep track of all label elements that are required
            self.required.add(element)
        return element

    def parse_group_element(self, group_info: Dict[str, any]) -> FuzzyTemplateGroupElement:
        """Parse a group element dictionary/JSON object into a fuzzy template group element.

        :param group_info: a dictionary containing the properties of the template group element
        :type group_info: dict
        :return: a fuzzy template group element
        :rtype: FuzzyTemplateGroupElement
        """
        group_elements = []
        for element_info in group_info["elements"]:
            if isinstance(element_info, str):
                # if element is a string, it must be a label element
                element_info = {"label": element_info, "type": "label"}
            elif "type" not in element_info:
                element_info["type"] = "group" if "elements" in element_info else "label"
            if element_info["type"] == "group":
                element = self.parse_group_element(element_info)
            else:
                element = self.parse_label_element(element_info)
                if element is None:
                    continue
            group_elements.append(element)
        group = generate_group_from_json(group_info, group_elements)
        # register the group element as part of the template
        self.groups.add(group)
        if group.label:
            self.group_element_index[group.label] = group
        return group
