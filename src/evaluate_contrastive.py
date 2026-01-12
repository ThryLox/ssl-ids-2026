import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import src.splits as splits
import src.models_contrastive as models_contrastive
import src.score as score

def main():
    # 1. Load data
    df = pd.read_csv('results/cleaned_sample.csv')
    X_train, X_val, X_test, y_test = splits.get_loato_splits(df, hold_out_family='DoS')
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Load model
    input_dim = X_train.shape[1]
    model = models_contrastive.ContrastiveEncoder(input_dim=input_dim)
    model.load_state_dict(torch.load("results/contrastive_encoder.pth"))
    model.eval()
    
    # 3. Get embeddings
    print("Extracting embeddings...")
    train_embeddings = score.get_model_embeddings(model, X_train_scaled)
    test_embeddings = score.get_model_embeddings(model, X_test_scaled)
    
    # 4. Fit Gaussian on train embeddings
    print("Fitting Mahalanobis distribution...")
    mu, inv_cov = score.fit_gaussian_embeddings(train_embeddings)
    
    # 5. Score test embeddings
    print("Scoring...")
    scores = score.score_mahalanobis(test_embeddings, mu, inv_cov)
    
    # 6. Evaluate
    auroc = roc_auc_score(y_test, scores)
    print(f"Contrastive SSL AUROC (DoS Hold-out): {auroc:.4f}")

if __name__ == "__main__":
    main()
