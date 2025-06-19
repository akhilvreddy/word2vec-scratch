import torch
from torch.utils.data import DataLoader
from word2vec.word2vec_vanilla import VanillaWord2Vec
from word2vec.word2vec_sgwns import SGNSWord2Vec
from word2vec.word2vec_cbow import CBOWWord2Vec
from dataset import Word2VecDataset
from utils import tokenize_corpus, build_vocab, generate_training_pairs

# Model selection
MODEL = "vanilla"

# Hyperparams
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.003
MIN_FREQ = 5
WINDOW_SIZE = 2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Can change to "mps" for Apple Silicon (m1/2/3/4 series)
# device = torch.device("mps" if torch.backends.mps.is_available() else device)

# Load + process data
tokens = tokenize_corpus("data/cleaned.txt")
word2idx, idx2word = build_vocab(tokens, min_freq=MIN_FREQ)
pairs = generate_training_pairs(tokens, word2idx, window_size=WINDOW_SIZE)

# Create dataset and dataloader
dataset = Word2VecDataset(pairs, word2idx)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model + loss + optimizer

if MODEL == "vanilla":
    model = VanillaWord2Vec(len(word2idx), EMBEDDING_DIM).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
elif MODEL == "sgns":
    model = SGNSWord2Vec(len(word2idx), EMBEDDING_DIM).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
elif MODEL == "cbow":
    model = CBOWWord2Vec(len(word2idx), EMBEDDING_DIM).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for center, context in dataloader:
        center = center.to(device)     # (B,)
        context = context.to(device)   # (B,)

        logits = model(center)         # (B, V)
        loss = loss_fn(logits, context)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")