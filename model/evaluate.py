import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix


def evaluate(y_test: np.array, y_pred: np.array):
    # evaluate
    metrics = {}
    threshold = 0.5

    y_pred_bin = np.where(y_pred > threshold, 1, 0)
    metrics['f1'] = f1_score(y_test, y_pred_bin)
    metrics['recall'] = recall_score(y_test, y_pred_bin)
    metrics['precision'] = precision_score(y_test, y_pred_bin)
    metrics['auc'] = roc_auc_score(y_test, y_pred)
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)

    print('Metric -- Value')
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"\t{k} : {round(v, 4)}")
        else:
            print(f"Confusion matrix:\n{v}")

    return metrics
