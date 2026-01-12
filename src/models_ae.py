import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_ae(model, train_loader, val_loader, epochs=20, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0]
                output = model(x)
                loss = criterion(output, x)
                val_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}")

def get_ae_scores(model, X):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X)
        output = model(x_tensor)
        reconstruction_error = torch.mean((output - x_tensor)**2, dim=1)
    return reconstruction_error.numpy()

if __name__ == "__main__":
    import src.splits as splits
    from sklearn.metrics import roc_auc_score
    
    df = pd.read_csv('results/cleaned_sample.csv')
    X_train, X_val, X_test, y_test = splits.get_loato_splits(df, hold_out_family='DoS')
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train_scaled))
    val_ds = TensorDataset(torch.FloatTensor(X_val_scaled))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    
    model = Autoencoder(input_dim=X_train.shape[1])
    print("Training Autoencoder...")
    train_ae(model, train_loader, val_loader, epochs=20)
    
    scores = get_ae_scores(model, X_test_scaled)
    print(f"Autoencoder AUROC: {roc_auc_score(y_test, scores):.4f}")
