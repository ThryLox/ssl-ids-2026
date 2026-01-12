import pandas as pd
from sklearn.model_selection import train_test_split

def get_loato_splits(df, hold_out_family):
    """
    Generator for Leave-One-Attack-Type-Out.
    Train on Normal only from the training split.
    Test contains Normal + held-out attack.
    """
    # Robust mapping
    def map_family(label):
        label = str(label)
        if 'BENIGN' in label: return 'Normal'
        if 'DoS' in label or 'DDoS' in label or 'Heartbleed' in label: return 'DoS'
        if 'PortScan' in label: return 'PortScan'
        if 'Bot' in label: return 'Bot'
        if 'Web Attack' in label: return 'WebAttack'
        if 'Patator' in label: return 'BruteForce'
        if 'Infiltration' in label: return 'Infiltration'
        return 'Other'

    df['Family'] = df['Label'].apply(map_family)
    
    # Normal data
    df_normal = df[df['Family'] == 'Normal']
    
    # Attack data
    df_attacks = df[df['Family'] != 'Normal']
    
    # Held-out attack family
    df_held_out = df_attacks[df_attacks['Family'] == hold_out_family]
    
    # Split Normal into Train/Val/Test
    train_normal, test_normal = train_test_split(df_normal, test_size=0.3, random_state=42)
    val_normal, test_normal = train_test_split(test_normal, test_size=0.5, random_state=42)
    
    # Training set: Normal only
    X_train = train_normal.drop(columns=['Label', 'Family'])
    
    # Test set: Normal from test_normal + Held-out attack
    X_test = pd.concat([test_normal, df_held_out]).drop(columns=['Label', 'Family'])
    y_test = (pd.concat([test_normal, df_held_out])['Family'] != 'Normal').astype(int)
    
    return X_train, val_normal.drop(columns=['Label', 'Family']), X_test, y_test

if __name__ == "__main__":
    df = pd.read_csv('results/cleaned_sample_full.csv')
    X_train, X_val, X_test, y_test = get_loato_splits(df, hold_out_family='DoS')
    print(f"LOATO for DoS:")
    print(f"Train Normal: {len(X_train)}")
    print(f"Test Set: {len(X_test)} (Anomaly Ratio: {y_test.mean():.2f})")
