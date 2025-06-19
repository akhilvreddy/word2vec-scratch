import torch
import torch.nn as nn

class VanillaWord2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)   # E_in: center word
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)  # E_out: output weights

    def forward(self, center_words):
        v = self.in_embed(center_words)         # (B, D)
        u = self.out_embed.weight               # (V, D)

        logits = v @ u.T           # (B, V)
        return logits  # raw logits (before softmax)

    def get_embedding(self, word_idx):
        return self.in_embed(word_idx)