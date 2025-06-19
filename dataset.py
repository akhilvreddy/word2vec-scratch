import torch
from torch.utils.data import Dataset
import random

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

class SGNSDataset(Dataset):
    def __init__(self, pairs, word2idx, negative_samples=5):
        self.pairs = [
            (word2idx[center], word2idx[context])
            for center, context in pairs
            if center in word2idx and context in word2idx
        ]
        self.word2idx = word2idx
        self.vocab_size = len(word2idx)
        self.negative_samples = negative_samples
        
        # unigram distribution for negative sampling
        word_freqs = {}
        for center, context in self.pairs:
            word_freqs[center] = word_freqs.get(center, 0) + 1
            word_freqs[context] = word_freqs.get(context, 0) + 1
        
        # convert to probabilities (3/4 power as in original word2vec)
        self.word_probs = {}
        total = sum(freq ** 0.75 for freq in word_freqs.values())
        for word, freq in word_freqs.items():
            self.word_probs[word] = (freq ** 0.75) / total
        
        self.words = list(self.word_probs.keys())
        self.probs = list(self.word_probs.values())

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, pos_word = self.pairs[idx]
        
        # generate negative samples
        neg_words = []
        for _ in range(self.negative_samples):
            neg_word = random.choices(self.words, weights=self.probs)[0]

            # Make sure negative sample is not the positive word (probably the most important / obvious thing to do)
            while neg_word == pos_word:
                neg_word = random.choices(self.words, weights=self.probs)[0]
            neg_words.append(neg_word)
        
        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(pos_word, dtype=torch.long),
            torch.tensor(neg_words, dtype=torch.long)
        )

class CBOWDataset(Dataset):
    def __init__(self, pairs, word2idx, negative_samples=5):
        self.pairs = pairs
        self.word2idx = word2idx
        self.vocab_size = len(word2idx)
        self.negative_samples = negative_samples
        
        # unigram distribution for negative sampling
        word_freqs = {}
        for context, center in self.pairs:
            word_freqs[center] = word_freqs.get(center, 0) + 1
            for ctx_word in context:
                word_freqs[ctx_word] = word_freqs.get(ctx_word, 0) + 1
        
        # Convert to probabilities (3/4 power as in original word2vec)
        self.word_probs = {}
        total = sum(freq ** 0.75 for freq in word_freqs.values())
        for word, freq in word_freqs.items():
            self.word_probs[word] = (freq ** 0.75) / total
        
        self.words = list(self.word_probs.keys())
        self.probs = list(self.word_probs.values())

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, center = self.pairs[idx]
        
        # Pad context to fixed size (use -1 as padding)
        max_context_size = max(len(pair[0]) for pair in self.pairs)
        if len(context) < max_context_size:
            context = context + [-1] * (max_context_size - len(context))
        
        # generate negative samples
        neg_words = []
        for _ in range(self.negative_samples):
            neg_word = random.choices(self.words, weights=self.probs)[0]
            
            # Make sure negative sample is not the center word
            while neg_word == center:
                neg_word = random.choices(self.words, weights=self.probs)[0]
            neg_words.append(neg_word)
        
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(center, dtype=torch.long),
            torch.tensor(neg_words, dtype=torch.long)
        )