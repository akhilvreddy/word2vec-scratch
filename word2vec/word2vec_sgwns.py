import torch
import torch.nn as nn
import torch.nn.functional as F

class SGNSWord2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)   # E_in: center
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)  # E_out: context

    def forward(self, center_words, pos_words, neg_words):
        """
        center_words: (B,)      # indices of center words
        pos_words:    (B,)      # indices of positive (context) words
        neg_words:    (B, K)    # indices of negative samples
        """

        v = self.in_embed(center_words)        # (B, D)
        u_pos = self.out_embed(pos_words)      # (B, D)
        u_neg = self.out_embed(neg_words)      # (B, K, D)

        # Positive score: (B,)
        pos_score = torch.sum(v * u_pos, dim=1)
        pos_loss = F.logsigmoid(pos_score)     # log σ(v · u_pos)

        # Negative scores: (B, K)
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # u_neg · v
        neg_loss = F.logsigmoid(-neg_score).sum(1)               # sum log σ(-v · u_neg)

        # Final loss (B,) → scalar
        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embedding(self, word_idx):
        return self.in_embed(word_idx)