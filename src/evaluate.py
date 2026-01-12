from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def evaluate_performance(y_true, y_scores):
    auroc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    return auroc, auprc

if __name__ == "__main__":
    print("Evaluation script initialized.")
