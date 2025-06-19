import torch
from torch.utils.data import Dataset

class Word2VecDataset(Dataset):
    def __init__(self, pairs, word2idx):
        self.pairs = [
            (word2idx[center], word2idx[context])
            for center, context in pairs
            if center in word2idx and context in word2idx
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)