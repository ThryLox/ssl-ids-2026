import pandas as pd
import torch
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import src.models_contrastive as models_contrastive
import src.score as score

def run_prototype(csv_path, model_path, scaler, mu, inv_cov, threshold):
    print(f"Ingesting flows from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Preprocess incoming flows (light cleaning)
    df_numeric = df.select_dtypes(include=[np.number])
    # Ensure columns match training
    # (In a real SOC, we'd have a fixed feature list)
    
    X_scaled = scaler.transform(df_numeric)
    
    # Load model and get embeddings
    input_dim = X_scaled.shape[1]
    model = models_contrastive.ContrastiveEncoder(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    embeddings = score.get_model_embeddings(model, X_scaled)
    scores = score.score_mahalanobis(embeddings, mu, inv_cov)
    
    alerts = df.copy()
    alerts['AnomalyScore'] = scores
    alerts['IsAnomaly'] = scores > threshold
    
    anomalous_flows = alerts[alerts['IsAnomaly'] == True]
    
    output_path = "results/alerts.csv"
    anomalous_flows.to_csv(output_path, index=False)
    print(f"Prototype finished. Alerts saved to {output_path}. Total Alerts: {len(anomalous_flows)}")

if __name__ == "__main__":
    print("Prototype script ready. Use run_prototype() in your pipeline.")
