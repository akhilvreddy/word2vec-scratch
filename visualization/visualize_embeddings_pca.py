import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import tokenize_corpus, build_vocab
from word2vec.word2vec_cbow import CBOWWord2Vec

# --- Config ---
MODEL_PATH = '../models/cbow_word2vec.pth'  # Change to your model path
DATA_PATH = '../data/cleaned.txt'           # Change to your data path
TOP_N = 200                                 # Number of words to visualize
EMBEDDING_DIM = 100                         # Should match your trained model

# --- Load vocab ---
tokens = tokenize_corpus(DATA_PATH)
word2idx, idx2word = build_vocab(tokens)

# --- Load model ---
model = CBOWWord2Vec(len(word2idx), EMBEDDING_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# --- Get embeddings ---
embeddings = model.in_embed.weight.data.cpu().numpy()

# --- Select top-N words ---
words = list(word2idx.keys())[:TOP_N]
indices = [word2idx[w] for w in words]
selected_embeddings = embeddings[indices]

# --- PCA ---
pca = PCA(n_components=2)
reduced = pca.fit_transform(selected_embeddings)

# --- Plot ---
plt.figure(figsize=(14, 12))
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
for i, word in enumerate(words):
    plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=9, alpha=0.7)
plt.title('Word Embeddings Visualization (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()