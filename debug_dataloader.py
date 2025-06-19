from torch.utils.data import DataLoader
from utils import tokenize_corpus, build_vocab, generate_training_pairs
from dataset import Word2VecDataset

tokens = tokenize_corpus("data/cleaned.txt")
word2idx, idx2word = build_vocab(tokens, min_freq=5)
pairs = generate_training_pairs(tokens, word2idx, window_size=2)

dataset = Word2VecDataset(pairs, word2idx)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    center, context = batch
    print("center:", center.shape)
    print("context:", context.shape)
    break