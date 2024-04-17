from typing import Any

from datasets import load_dataset
from torch.utils.data import Dataset

def load_data(splitting: str) -> tuple[list[str], list[int]]:
    """
    Loads the example data, the Yelp review dataset, and returns the text inputs with corresponding labels for either the training or test split.

    Args:
        splitting (str): A string indicating which split of the dataset to load. Expected values are 'train' for the training set and any other value will default to loading the test set.

    Returns:
        tuple[list[str], list[int]]: A tuple containing two lists:
            - The first list contains the text of the reviews.
            - The second list contains the corresponding labels as integers.
    """
    dataset = load_dataset("yelp_review_full")
    if splitting == 'train':
        train_input = dataset['train']['text']
        train_labels = dataset['train']['label']
        return train_input, train_labels
    test_input = dataset['test']['text']
    test_labels = dataset['test']['label']
    return test_input, test_labels

class ExampleDataset(Dataset):
    def __init__(self, splitting: str) -> None:
        inputs, labels = load_data(splitting)
        self.data = list(zip(inputs, labels))
        super().__init__()

    def __getitem__(self, index: int) -> Any:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
