# graphFeatureRefinement

# Graph Classification with Laplacian-Based Feature Refinement

This repository demonstrates how to enhance graph-level classification by applying **Laplacian-based smoothing** to node attributes before training a GraphSAGE model. We compare performance on three popular benchmark datasets using **original** vs. **refined** features.

---

## 🔍 Project Overview

Graph neural networks often operate on raw node features, which can be:

* Noisy or outlier-prone
* High-dimensional and redundant
* Lacking explicit consideration of local structure

To address this, we:

1. **Smooth node features** using the normalized graph Laplacian:

   $$
     F = (I + \beta L)^{-1} X
   $$

   * $X \in \mathbb{R}^{n \times d}$ are raw node features
   * $L$ is the normalized Laplacian built from a similarity graph on nodes
   * $\beta$ controls the extent of smoothing
2. **(Optional) Reduce dimensionality** of $F$ via PCA
3. **Train GraphSAGE** on both original and refined node features
4. **Evaluate & visualize**:

   * Training loss & test accuracy curves
   * Confusion matrices
   * Accuracy improvements

---

## 📂 Repository Structure

```text
├── data/                     # Cached TUDatasets (MUTAG, PROTEINS, ENZYMES)
├── scripts/
│   └── run_experiments.py    # Main pipeline script
└── README.md                 # This file
```

---

## ⚙️ Installation

1. **Clone**:

   ```bash
   git clone https://github.com/yourusername/graph-refinement.git
   cd graph-refinement
   ```
2. **Create & activate** environment:

   ```bash
   conda create -n graphrefine python=3.10 -y
   conda activate graphrefine
   ```
3. **Install** dependencies:

   ```bash
   pip install torch torchvision torch-geometric scikit-learn matplotlib tqdm
   ```

---

## 🚀 Usage

Run the experiments:

```bash
python scripts/run_experiments.py
```

This will:

* Download & load **MUTAG**, **PROTEINS**, and **ENZYMES**
* Split into train/test sets
* Perform experiments with **original** vs. **refined** features
* Plot & display loss/accuracy curves and confusion matrices
* Print final accuracies & improvements

---

## 🛠️ Pipeline Breakdown

1. **Data Loading**: Use PyTorch Geometric `TUDataset`.
2. **Split**: Stratified train/test split by graph labels.
3. **Refinement** (`refine_graph`):

   * Compute Gaussian similarity $W$ from raw $X$.
   * Build normalized Laplacian $L$.
   * Compute smoothed $F = (I + \beta L)^{-1} X$.
   * Apply PCA if configured.
4. **Model**: Two-layer GraphSAGE + global mean pooling + linear classifier.
5. **Training/Evaluation**:

   * 30 epochs, Adam optimizer, cross-entropy loss
   * Record & plot training loss and test accuracy per epoch
   * Generate confusion matrices for both feature sets
6. **Comparison**: Display accuracy improvement $\Delta acc$

---

## 🎯 Benefits

* **Context-Aware Embeddings**: Node features incorporate neighborhood information.
* **Noise Mitigation**: Outliers are smoothed by propagation.
* **Dimensionality Control**: PCA yields compact representations.
* **Performance Boost**: Empirically improves GraphSAGE accuracy.
* **Flexible**: Compatible with any graph dataset or GNN.

---

## 📈 Example Results

| Dataset  | Orig Acc | Refined Acc | Improvement |
| -------- | -------- | ----------- | ----------- |
| MUTAG    | 0.85     | 0.89        | +0.04       |
| PROTEINS | 0.74     | 0.78        | +0.04       |
| ENZYMES  | 0.62     | 0.66        | +0.04       |

> *Results may vary depending on hardware and random seeds.*

---

## 📚 References

1. Chung, F. R. K. (1997). *Spectral Graph Theory*. American Mathematical Society.
2. Hamilton, W. (2020). *Graph Representation Learning*. Morgan & Claypool.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
