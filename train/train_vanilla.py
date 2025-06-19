import torch
from torch.utils.data import DataLoader
from word2vec.word2vec_vanilla import VanillaWord2Vec
from dataset import Word2VecDataset
from utils import tokenize_corpus, build_vocab, generate_training_pairs

# Hyperparams
EMBEDDING_DIM = 100 # try larger to capture more more semantic relationships
BATCH_SIZE = 128 # try smaller to capture more local relationships
EPOCHS = 5 # experiment with 1, 5, 10, 20 to see how it affects results
LR = 0.003 # Implementing a LR scheduler should give us faster convergence / similar results
MIN_FREQ = 5 # you need minimum 3-5 for good results (depends on your corpus)
WINDOW_SIZE = 2 # I like 2-3 for good results (also depends on how biased / skewed your corpus is)

# Device
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

def train_vanilla():
    print("Training Vanilla Word2Vec...")
    
    # Load + process data
    tokens = tokenize_corpus("data/cleaned.txt")
    word2idx, idx2word = build_vocab(tokens, min_freq=MIN_FREQ)
    pairs = generate_training_pairs(tokens, word2idx, window_size=WINDOW_SIZE)
    
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Training pairs: {len(pairs)}")
    
    # Create dataset and dataloader
    dataset = Word2VecDataset(pairs, word2idx)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model + loss + optimizer
    model = VanillaWord2Vec(len(word2idx), EMBEDDING_DIM).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = 0
        
        for center, context in dataloader:
            center = center.to(device)     # (B,)
            context = context.to(device)   # (B,)
            
            logits = model(center)         # (B, V)
            loss = loss_fn(logits, context)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/vanilla_word2vec.pth')
    print("Training completed! Model saved to models/vanilla_word2vec.pth")

if __name__ == "__main__":
    train_vanilla() 