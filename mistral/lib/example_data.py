from typing import Any

from datasets import load_dataset
from torch.utils.data import Dataset

def load_data(splitting: str) -> tuple[list[str], list[int]]:
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
