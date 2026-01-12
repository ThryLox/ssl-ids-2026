import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import src.splits as splits
import src.models_contrastive as models_contrastive
import src.score as score
import src.models_ae as models_ae
import src.baselines_rules as baselines_rules
import src.baselines_classical as baselines_classical
from sklearn.metrics import roc_auc_score
import os

def run_experiment(df, hold_out_family, aug_types):
    X_train, X_val, X_test, y_test = splits.get_loato_splits(df, hold_out_family=hold_out_family)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Train Contrastive SSL with specified augmentations
    dataset = models_contrastive.ContrastiveDataset(X_train_scaled)
    # Patch augment function for ablation
    def custom_augment(x):
        res = x.clone()
        if 'jitter' in aug_types:
            res = res + torch.randn_like(res) * 0.02
        if 'masking' in aug_types:
            mask = torch.rand_like(res) > 0.1
            res = res * mask
        if 'scaling' in aug_types:
            scale = torch.empty(1).uniform_(0.9, 1.1)
            res = res * scale
        return res
    
    dataset.augment = custom_augment
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = models_contrastive.ContrastiveEncoder(input_dim=X_train.shape[1])
    models_contrastive.train_contrastive(model, loader, epochs=20)
    
    train_embeddings = score.get_model_embeddings(model, X_train_scaled)
    test_embeddings = score.get_model_embeddings(model, X_test_scaled)
    mu, inv_cov = score.fit_gaussian_embeddings(train_embeddings)
    ssl_scores = score.score_mahalanobis(test_embeddings, mu, inv_cov)
    ssl_auroc = roc_auc_score(y_test, ssl_scores)
    
    return ssl_auroc

def main():
    df = pd.read_csv('results/cleaned_sample.csv')
    families = ['DoS', 'Infiltration', 'WebAttack']
    ablations = [
        ['jitter'],
        ['jitter', 'masking'],
        ['jitter', 'masking', 'scaling']
    ]
    
    results = []
    for family in families:
        for aug in ablations:
            print(f"--- Running Family: {family}, Aug: {aug} ---")
            auroc = run_experiment(df, family, aug)
            results.append({
                'Family': family,
                'Augmentations': "+".join(aug),
                'AUROC': auroc
            })
    
    res_df = pd.DataFrame(results)
    print("\nAblation Results:")
    print(res_df)
    res_df.to_csv("results/ablation_results.csv", index=False)

if __name__ == "__main__":
    main()
