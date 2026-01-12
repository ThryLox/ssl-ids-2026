from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def train_isolation_forest(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled)
    return model, scaler

if __name__ == "__main__":
    import src.splits as splits
    from sklearn.metrics import roc_auc_score
    
    df = pd.read_csv('results/cleaned_sample.csv')
    X_train, X_val, X_test, y_test = splits.get_loato_splits(df, hold_out_family='DoS')
    
    print("Training Isolation Forest...")
    model, scaler = train_isolation_forest(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    scores = -model.decision_function(X_test_scaled) # Negative because decision_function returns higher for normal
    
    print(f"Isolation Forest AUROC: {roc_auc_score(y_test, scores):.4f}")
