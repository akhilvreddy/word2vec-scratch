import re
from collections import Counter

def tokenize_corpus(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    tokens = text.strip().split()
    return tokens

def build_vocab(tokens, min_freq=5):
    counter = Counter(tokens)
    filtered = {word: freq for word, freq in counter.items() if freq >= min_freq}

    word2idx = {word: idx for idx, word in enumerate(filtered)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word

def generate_training_pairs(tokens, word2idx, window_size=2):
    pairs = []

    for center_idx in range(len(tokens)):
        center_word = tokens[center_idx]

        if center_word not in word2idx:
            continue

        for offset in range(-window_size, window_size + 1):
            context_idx = center_idx + offset
            if offset == 0 or context_idx < 0 or context_idx >= len(tokens):
                continue

            context_word = tokens[context_idx]
            if context_word not in word2idx:
                continue

            pairs.append((center_word, context_word))

    return pairs

def generate_cbow_pairs(tokens, word2idx, window_size=2):
    pairs = []
    for sentence in tokens:
        for i in range(len(sentence)):
            center = word2idx.get(sentence[i], None)
            if center is None:
                continue

            context = []
            for j in range(i - window_size, i + window_size + 1):
                if j != i and 0 <= j < len(sentence):
                    ctx_word = word2idx.get(sentence[j], None)
                    if ctx_word is not None:
                        context.append(ctx_word)

            if len(context) > 0:
                pairs.append((context, center))
    return pairs