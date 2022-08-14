import torch
from torch.utils.data import Dataset

from src.utils.helpers import cents_to_bins, freqs_to_cents


class DDSPDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.features = torch.load(path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            cents_to_bins(freqs_to_cents(self.features[idx]["f0"])) / 359,
            (self.features[idx]["loudness"] - 44.0666) / 5.468337,
            self.features[idx]["audio"],
        )
