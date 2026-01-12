import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class ContrastiveDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def augment(self, x):
        # 1. Gaussian jitter
        x_jitter = x + torch.randn_like(x) * 0.02
        
        # 2. Feature masking
        mask = torch.rand_like(x) > 0.1
        x_masked = x_jitter * mask
        
        # 3. Scaling
        scale = torch.empty(1).uniform_(0.9, 1.1)
        x_scaled = x_masked * scale
        
        return x_scaled

    def __getitem__(self, idx):
        x = self.data[idx]
        return self.augment(x), self.augment(x)

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super(ContrastiveEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

def nt_xent_loss(z_i, z_j, temperature=0.2):
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    
    sim_ij = torch.diag(sim_matrix, batch_size)
    sim_ji = torch.diag(sim_matrix, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)
    
    mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    negatives = sim_matrix[mask].view(2 * batch_size, -1)
    
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    logits /= temperature
    
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)

def train_contrastive(model, loader, epochs=20, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for x_i, x_j in loader:
            optimizer.zero_grad()
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            loss = nt_xent_loss(z_i, z_j)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Contrastive Loss: {total_loss/len(loader):.6f}")

if __name__ == "__main__":
    import src.splits as splits
    
    df = pd.read_csv('results/cleaned_sample.csv')
    X_train, X_val, X_test, y_test = splits.get_loato_splits(df, hold_out_family='DoS')
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    dataset = ContrastiveDataset(X_train_scaled)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = ContrastiveEncoder(input_dim=X_train.shape[1])
    print("Training Contrastive Encoder...")
    train_contrastive(model, loader, epochs=20)
    
    # Next step: Scoring with Mahalanobis in score.py
    torch.save(model.state_dict(), "results/contrastive_encoder.pth")
    print("Model saved.")
