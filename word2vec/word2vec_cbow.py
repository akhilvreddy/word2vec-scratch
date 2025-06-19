import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOWWord2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)   # E_in: center
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)  # E_out: context

    def forward(self, context_words, center_words, neg_words):
        """
        context_words: (B, C)   # indices of context words
        center_words: (B,)      # indices of center words
        neg_words:    (B, K)    # indices of negative samples
        """

        context_embeddings = self.in_embed(context_words)
        context_embeddings = context_embeddings.mean(dim=1) # take mean of context words for each center word

        center_embedding = self.out_embed(center_words)
        neg_embedding = self.out_embed(neg_words)

        pos_score = torch.sum(context_embeddings * center_embedding, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_embedding, context_embeddings.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(1)

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embedding(self, word_idx):
        return self.in_embed(word_idx)