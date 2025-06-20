# PCA and t-SNE Visualizations

This directory contains scripts to visualize word embeddings using both PCA and t-SNE. These visualizations help you understand the structure of your learned embeddings.

## How to Use

1. Make sure you have trained your word2vec model and saved the embeddings (to your /models folder).
2. Run the visualization scripts:
   ```bash
   python visualize_embeddings_pca.py
   python visualize_embeddings_tsne.py
   ```
3. The scripts will display (or save) 2D scatter plots of your word embeddings.

## Core Differences

| Method | What it does | Best for | Limitations |
|--------|--------------|----------|-------------|
| **PCA** | Finds the main axes of variance in your data (linear) | Understanding overall structure, global relationships | May not reveal clusters if data is nonlinear |
| **t-SNE** | Preserves local neighborhoods, reveals clusters (nonlinear) | Visualizing clusters, local relationships | Results can vary between runs, global distances not meaningful |

### PCA

PCA helps you see the main axes that affect your data. It's good for understanding the corpus holistically, but words that share variance along the main axes may appear close even if they're not semantically related.

### t-SNE

t-SNE is great for visualizing clusters and local relationships (e.g., synonyms, gendered word forms). However, it's not stable—results can differ between runs. (Setting a random seed can help with reproducibility.)

## Dependencies

- matplotlib
- scikit-learn
- torch (for loading embeddings)

(Covered in requirements.txt so don't worry if you already pip installed)

## Tips

- Look for clusters of semantically similar words.
- Try both PCA and t-SNE to get different perspectives on your embeddings.

---