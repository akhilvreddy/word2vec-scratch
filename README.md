# Word2Vec Implementation from Scratch

This repository contains a PyTorch implementation of the Word2Vec model, including both CBOW (Continuous Bag of Words) and Skip-gram architectures with Negative Sampling. This is the accompanying code for my [blog post](https://www.google.com).

## 🚀 Features

- Multiple Word2Vec implementations:
  - Vanilla Word2Vec
  - Skip-gram with Negative Sampling (SGNS)
  - Continuous Bag of Words (CBOW)
- Training on wikipedia text corpus
- Efficient data processing and batching
- Visualizations for final word embeddings
- Support for CPU and GPU (CUDA/MPS) training \\
  (I just used CPU but have tagged the sections you can use parallelization)

## 📋 Requirements

- Python 3.10 or above
- PyTorch 2.0+
- [Optional] CUDA (for GPU support)
- [Optional] Apple Silicon (for MPS support)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/akhilvreddy/word2vec-scratch.git
cd word2vec-scratch
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Project Structure

```
word2vec-scratch/
├── data/                    # Data directory (created after running load_data.py)
├── word2vec/               
│   ├── word2vec_vanilla.py  # Vanilla Word2Vec implementation
│   ├── word2vec_sgns.py    # Skip-gram with negative sampling
│   └── word2vec_cbow.py    # Continuous Bag of Words implementation
├── utils.py                 # Utility functions for data processing
├── dataset.py              # PyTorch dataset implementations
├── train.py                # Training script (for all methods)
├── load_data.py            # Data preparation script
└── requirements.txt        # Project dependencies
```

## 🚦 Getting Started

1. Prepare the training data:
```bash
python load_data.py
```

2. Train the model:
```bash
python train.py
```

You can modify the hyperparameters in `train.py`:
- `EMBEDDING_DIM`: Dimension of word embeddings
- `BATCH_SIZE`: Number of samples per batch
- `EPOCHS`: Number of training epochs
- `LR`: Learning rate
- `MIN_FREQ`: Minimum word frequency for vocabulary
- `WINDOW_SIZE`: Context window size
- `MODEL`: Choose between "vanilla", "sgns", or "cbow"

(I'll add argparse support later, sorry about that. And I'll add wandb logging too).

## 💡 Usage Example

```python
import torch
from word2vec.word2vec_cbow import CBOWWord2Vec

# Load trained model
model = CBOWWord2Vec(vocab_size=10000, embedding_dim=100)
model.load_state_dict(torch.load('model.pth'))

# Get word embeddings
word_embedding = model.get_embedding(word_idx)
```

## 📈 Results

Training results, loss curves, and example word similarities coming soon.

## 📚 References

- Mikolov, T., et al. (2013). [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Mikolov, T., et al. (2013). [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) 