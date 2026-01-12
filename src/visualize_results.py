import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import torch
import os
import src.splits as splits
import src.models_contrastive as models_contrastive
import src.score as score
import src.baselines_rules as baselines_rules
import src.baselines_classical as baselines_classical

def plot_roc_curves(y_test, scores_dict, family_name):
    plt.figure(figsize=(10, 6))
    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {family_name} Detection')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(f'results/roc_{family_name}.png')
    plt.close()

def plot_tsne(embeddings, labels, family_name):
    print(f"Generating t-SNE for {family_name}...")
    # Sampling for speed if needed
    if len(embeddings) > 5000:
        idx = np.random.choice(len(embeddings), 5000, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=z_tsne[:, 0], y=z_tsne[:, 1], hue=labels, palette='viridis', alpha=0.6)
    plt.title(f't-SNE Projection of SSL Embeddings ({family_name})')
    plt.legend(title='Category')
    plt.savefig(f'results/tsne_{family_name}.png')
    plt.close()

def main():
    os.makedirs('results', exist_ok=True)
    df = pd.read_csv('results/cleaned_sample_full.csv')
    
    # We'll visualize for "DoS" as the primary example
    family = 'DoS'
    print(f"Visualizing for family: {family}")
    
    X_train, X_val, X_test, y_test = splits.get_loato_splits(df, hold_out_family=family)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Get Baseline scores
    # Rule based
    ids = baselines_rules.RuleIDS()
    ids.fit(X_train)
    rule_scores = ids.predict_score(X_test)
    
    # Isolation Forest
    if_model, _ = baselines_classical.train_isolation_forest(X_train)
    if_scores = -if_model.decision_function(X_test_scaled)
    
    # 2. Get Contrastive SSL scores
    model = models_contrastive.ContrastiveEncoder(input_dim=X_train.shape[1])
    dataset = models_contrastive.ContrastiveDataset(X_train_scaled)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    print("Training visualizing model...")
    models_contrastive.train_contrastive(model, loader, epochs=10) # Quick training for visualization
    
    train_embeddings = score.get_model_embeddings(model, X_train_scaled)
    test_embeddings = score.get_model_embeddings(model, X_test_scaled)
    mu, inv_cov = score.fit_gaussian_embeddings(train_embeddings)
    ssl_scores = score.score_mahalanobis(test_embeddings, mu, inv_cov)
    
    # 3. Plot ROC
    scores_dict = {
        'Rule-based': rule_scores,
        'Isolation Forest': if_scores,
        'Contrastive SSL': ssl_scores
    }
    plot_roc_curves(y_test, scores_dict, family)
    
    # 4. Plot t-SNE
    # Label categorical names for plotting
    test_labels = np.where(y_test == 1, family, 'Normal')
    plot_tsne(test_embeddings, test_labels, family)
    
    print("Visualizations complete. Files saved in results/.")

if __name__ == "__main__":
    main()
