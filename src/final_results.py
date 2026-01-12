import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import src.splits as splits
import src.models_contrastive as models_contrastive
import src.score as score
import src.models_ae as models_ae
import src.baselines_rules as baselines_rules
import src.baselines_classical as baselines_classical
import json
import os

from sklearn.metrics import roc_curve, auc

def get_tpr_at_fpr(y_true, y_scores, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Find the TPR where FPR is closest to target_fpr
    idx = np.argmin(np.abs(fpr - target_fpr))
    return tpr[idx]

def main():
    # 1. Load data
    DATA_PATH = 'results/cleaned_sample_full.csv'
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run preprocess.py first.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # Dataset Statistics Table
    print("\n=== Dataset Statistics ===")
    stats = df['Label'].value_counts()
    print(stats)
    
    families = ['DoS', 'PortScan', 'WebAttack', 'BruteForce']
    final_results = {}

    for family in families:
        print(f"\n=== Evaluating Family: {family} ===")
        # Note: splits.get_loato_splits already separates normal and attack
        # Train and Val contain only Normal. Test contains Normal + Attack (Family)
        X_train, X_val, X_test, y_test = splits.get_loato_splits(df, hold_out_family=family)
        
        if len(y_test.unique()) < 2:
            print(f"Warning: Only one class in test set for {family}. Skipping.")
            continue
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Calibration Protocol: Thresholds selected on normal validation traffic
        # target_fpr = 0.01 (1%)
        
        # 1. Rule-based
        ids = baselines_rules.RuleIDS()
        ids.fit(X_train)
        rule_scores = ids.predict_score(X_test)
        rule_auroc = roc_auc_score(y_test, rule_scores)
        rule_tpr = get_tpr_at_fpr(y_test, rule_scores, 0.01)
        
        # 2. Isolation Forest
        if_model, _ = baselines_classical.train_isolation_forest(X_train)
        if_scores = -if_model.decision_function(X_test_scaled)
        if_auroc = roc_auc_score(y_test, if_scores)
        if_tpr = get_tpr_at_fpr(y_test, if_scores, 0.01)
        
        # 3. Contrastive SSL
        model = models_contrastive.ContrastiveEncoder(input_dim=X_train.shape[1])
        dataset = models_contrastive.ContrastiveDataset(X_train_scaled)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        models_contrastive.train_contrastive(model, loader, epochs=20)
        
        # Representation extraction
        train_embeddings = score.get_model_embeddings(model, X_train_scaled)
        val_embeddings = score.get_model_embeddings(model, X_val_scaled)
        test_embeddings = score.get_model_embeddings(model, X_test_scaled)
        
        # Fit Gaussian on Train (Normal only)
        mu, inv_cov = score.fit_gaussian_embeddings(train_embeddings)
        
        # Scoring
        ssl_scores = score.score_mahalanobis(test_embeddings, mu, inv_cov)
        ssl_auroc = roc_auc_score(y_test, ssl_scores)
        ssl_tpr = get_tpr_at_fpr(y_test, ssl_scores, 0.01)
        
        final_results[family] = {
            'AUROC': {
                'Rule-based': rule_auroc,
                'IsolationForest': if_auroc,
                'ContrastiveSSL': ssl_auroc
            },
            'TPR@1%FPR': {
                'Rule-based': rule_tpr,
                'IsolationForest': if_tpr,
                'ContrastiveSSL': ssl_tpr
            }
        }

    with open("results/final_refined_metrics.json", "w") as f:
        json.dump(final_results, f, indent=4)
    
    print("\nFinal Refined Results Summary:")
    print(json.dumps(final_results, indent=4))

if __name__ == "__main__":
    main()
