import torch
from torch.utils.data import DataLoader
from word2vec.word2vec_cbow import CBOWWord2Vec
from dataset import CBOWDataset
from utils import tokenize_corpus, build_vocab, generate_cbow_pairs

# Hyperparams
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.003
MIN_FREQ = 5
WINDOW_SIZE = 2
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive pair

# Device
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

def train_cbow():
    print("Training Continuous Bag of Words (CBOW)...")
    
    # Load + process data
    tokens = tokenize_corpus("data/cleaned.txt")
    word2idx, idx2word = build_vocab(tokens, min_freq=MIN_FREQ)
    
    # Generate CBOW-specific pairs (context -> center)
    pairs = generate_cbow_pairs(tokens, word2idx, window_size=WINDOW_SIZE)
    
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Training pairs: {len(pairs)}")
    
    # Create dataset and dataloader
    dataset = CBOWDataset(pairs, word2idx, negative_samples=NEGATIVE_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model + optimizer (no external loss function needed)
    model = CBOWWord2Vec(len(word2idx), EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = 0
        
        for context_words, center_words, neg_words in dataloader:
            context_words = context_words.to(device)  # (B, C)
            center_words = center_words.to(device)    # (B,)
            neg_words = neg_words.to(device)          # (B, K)
            
            # Model computes loss internally
            loss = model(context_words, center_words, neg_words)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/cbow_word2vec.pth')
    print("Training completed! Model saved to models/cbow_word2vec.pth")

if __name__ == "__main__":
    train_cbow() 