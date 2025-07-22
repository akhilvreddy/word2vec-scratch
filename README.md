This is the accompanying repository for my [blog post](https://akhilvreddy.com/posts/word2vec-scratch/). Please check out [this notebook](https://github.com/akhilvreddy/word2vec-scratch/blob/main/notebooks/Word2Vec_Insights.ipynb) for the important code.

---

# Word2Vec Implementation from Scratch

This repository contains a PyTorch implementation of the Word2Vec model, including both CBOW (Continuous Bag of Words) and Skip-gram architectures with Negative Sampling. This is the accompanying code for my blog post.

## Features

- Multiple Word2Vec implementations:
  - Vanilla Word2Vec
  - Skip-gram with Negative Sampling (SGNS)
  - Continuous Bag of Words (CBOW)

> My loss calculations for these happen inside the `forward` function for each class instead of the training loop itself (`loss = model(inputs)`).

- Training on wikipedia text corpus
- Visualizations for final word embeddings
- Support for CPU and GPU (CUDA/MPS) training

## Requirements

- Python 3.10 or above
- PyTorch 2.0+

## Installation

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

You can modify the hyperparameters in each of the training files:
- `EMBEDDING_DIM`: Dimension of word embeddings
- `BATCH_SIZE`: Number of samples per batch
- `EPOCHS`: Number of training epochs
- `LR`: Learning rate
- `MIN_FREQ`: Minimum word frequency for vocabulary
- `WINDOW_SIZE`: Context window size
- `MODEL`: Choose between "vanilla", "sgns", or "cbow"


## Usage

```python
import torch
from word2vec.word2vec_cbow import CBOWWord2Vec

# Load trained model
model = CBOWWord2Vec(vocab_size=10000, embedding_dim=100)
model.load_state_dict(torch.load('model.pth'))

# Get word embeddings
word_embedding = model.get_embedding(word_idx)
```

## 📚 References

- Mikolov, T., et al. (2013). [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Mikolov, T., et al. (2013). [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) 
