# Manifold-Based Novelty Detection for Network Intrusion

![t-SNE Projection](results/tsne_DoS.png)

## Overview
This repository contains the implementation and research paper for a **Contrastive Self-Supervised Learning (SSL)** approach to Network Intrusion Detection (NIDS). By learning the "Manifold of Normalcy" without labels, this system is capable of detecting novel, zero-day attacks by measuring their statistical displacement from learned behavioral anchors.

## Key Features
- **Behavioral Manifold**: Learns robust representations of benign traffic using InfoNCE loss.
- **Zero-Day Generalization**: Evaluated using Leave-One-Attack-Type-Out (LOATO) protocols.
- **Operational Nuance**: Benchmarked under strict 1% FPR operational budgets, reflecting real-world SOC constraints.
- **Explainability**: Anomaly score attribution via Mahalanobis distance decomposition.

## Repository Structure
- `/src`: PyTorch source code for the SSL encoder, preprocessing, and evaluation.
- `/results`: Visualizations (t-SNE, ROC) and metrics.
- `/data`: (Excluded from Git) Training data source scripts.

## Getting Started
1. **Prerequisites**: Python 3.9+, PyTorch 2.0+.
2. **Setup**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Training**:
   Run `src/models_contrastive.py` to train the encoder on benign traffic.
4. **Evaluation**:
   Use `src/evaluate_contrastive.py` to perform the LOATO analysis.

## Citation
If you find this research useful, please cite:
```
@article{singh2024manifold,
  title={Manifold-Based Novelty Detection for Network Intrusion: A Contrastive Self-Supervised Approach},
  author={Singh, Ekonkar},
  year={2024}
}
```
