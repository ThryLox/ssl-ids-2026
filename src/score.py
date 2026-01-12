import numpy as np
import torch
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import LedoitWolf

def fit_gaussian_embeddings(embeddings):
    """
    Fit a Gaussian on embeddings using Ledoit-Wolf shrinkage for covariance stability.
    """
    mu = np.mean(embeddings, axis=0)
    lw = LedoitWolf().fit(embeddings)
    inv_cov = lw.precision_
    return mu, inv_cov

def score_mahalanobis(embeddings, mu, inv_cov):
    """
    Compute Mahalanobis distance for all embeddings.
    """
    diff = embeddings - mu
    # Efficiently compute (diff @ inv_cov @ diff.T)
    scores = np.sqrt(np.sum(np.dot(diff, inv_cov) * diff, axis=1))
    return scores

def get_model_embeddings(model, X_scaled, device='cpu'):
    """
    Extract latent representations (h) from the ContrastiveEncoder.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_scaled).to(device)
        h, _ = model(x_tensor)
    return h.cpu().numpy()

if __name__ == "__main__":
    print("Scoring module updated.")
