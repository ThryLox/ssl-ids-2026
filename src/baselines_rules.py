import numpy as np
import pandas as pd

class RuleIDS:
    def __init__(self, percentile=99.9):
        self.percentile = percentile
        self.thresholds = {}

    def fit(self, X_train):
        """
        Calculate thresholds for each feature based on the training normal data.
        """
        for column in X_train.columns:
            self.thresholds[column] = np.percentile(X_train[column], self.percentile)
        print(f"RuleIDS trained on {len(X_train.columns)} features.")

    def predict_score(self, X):
        """
        Anomaly score = number of features exceeding their 99.9th percentile threshold.
        """
        # Count how many features exceed the threshold
        scores = np.zeros(len(X))
        for col, threshold in self.thresholds.items():
            if col in X.columns:
                scores += (X[col] > threshold).astype(int)
        return scores

    def predict(self, X, threshold_count=1):
        """
        Flag as anomaly if at least 'threshold_count' features exceed their thresholds.
        """
        scores = self.predict_score(X)
        return (scores >= threshold_count).astype(int)

if __name__ == "__main__":
    import src.splits as splits
    df = pd.read_csv('results/cleaned_sample.csv')
    X_train, X_val, X_test, y_test = splits.get_loato_splits(df, hold_out_family='DoS')
    
    ids = RuleIDS()
    ids.fit(X_train)
    scores = ids.predict_score(X_test)
    
    from sklearn.metrics import roc_auc_score
    print(f"RuleIDS AUROC: {roc_auc_score(y_test, scores):.4f}")
