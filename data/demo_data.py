from typing import Dict, List

from data import auction_advertisements


datasets = {
    "auction_advertisements": {
        "template": auction_advertisements.auction_template,
        "phrases": auction_advertisements.auction_phrases,
        "texts": auction_advertisements.auction_texts,
        "tests": auction_advertisements.auction_tests
    }
}


def unknown_dataset_name(dataset_name: str, known_datasets: List[str]) -> None:
    raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets are: {known_datasets}")


class DemoData:

    def __init__(self):
        """A object for accessing datasets for demonstrations and testing. It contains the following datasets:
        1. auction_advertisements: this is a use case of digitized 18th century Dutch newspapers from the National
        Library of the Netherlands. It contains a small sample of texts from newspaper advertisements of auctions,
        as well as a set of phrases for common elements in those advertisements, and a template for describing
        how the phrases constitute the various elements of an auction advertisement, such that the fuzzy searcher
        can identify those elements in advertisement texts, even though the character error rate is very high.
        """
        self.datasets = datasets
        self.templates = {dataset_name: datasets[dataset_name]["template"] for dataset_name in datasets}
        self.phrases = {dataset_name: datasets[dataset_name]["phrases"] for dataset_name in datasets}
        self.texts = {dataset_name: datasets[dataset_name]["texts"] for dataset_name in datasets}

    def get_dataset(self, dataset_name: str) -> Dict[str, any]:
        """Return the dataset for a given dataset name.

        :param dataset_name: the name of a dataset included in the fuzzy-search package
        :type dataset_name: str
        :return: a dataset including texts, phrases and templates
        :rtype: dict
        """
        if dataset_name not in self.datasets:
            unknown_dataset_name(dataset_name, list(self.datasets.keys()))
        return self.datasets[dataset_name]

    def get_phrases(self, dataset_name: str) -> List[Dict[str, any]]:
        """Return the phrases for a given dataset.

        :param dataset_name: the name of a dataset included in the fuzzy-search package
        :type dataset_name: str
        :return: a list of phrase dictionaries of the given dataset
        :rtype: List[Dict[str, any]]
        """
        if dataset_name not in self.phrases:
            unknown_dataset_name(dataset_name, list(self.datasets.keys()))
        return self.phrases[dataset_name]

    def get_template(self, dataset_name: str) -> Dict[str, any]:
        """Return the template for a given dataset.

        :param dataset_name: the name of a dataset included in the fuzzy-search package
        :type dataset_name: str
        :return: a list of template elements of the given dataset
        :rtype: List[Dict[str, any]]
        """
        if dataset_name not in self.templates:
            unknown_dataset_name(dataset_name, list(self.datasets.keys()))
        return self.templates[dataset_name]

    def get_texts(self, dataset_name: str) -> List[str]:
        """Return the texts for a given dataset.

        :param dataset_name: the name of a dataset included in the fuzzy-search package
        :type dataset_name: str
        :return: a list of texts of the given dataset
        :rtype: List[str]
        """
        if dataset_name not in self.texts:
            unknown_dataset_name(dataset_name, list(self.datasets.keys()))
        return self.texts[dataset_name]

