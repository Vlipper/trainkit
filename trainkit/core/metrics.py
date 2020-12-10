import numpy as np
from sklearn.metrics import roc_auc_score


def custom_ova_roc_auc(predict: np.ndarray,
                       target: np.ndarray) -> float:
    """Calculates multi-class ROC AUC (one-vs-all)

    Args:
        predict: scores array with shape (num_samples, num_classes)
        target: targets array with shape (num_samples,). Where `0 <= target[i] < num_classes`

    Returns:
        Mean value of class-wise ROC AUC score
    """
    roc_auc_per_label, num_label = [], len(np.unique(target))

    for i in range(num_label):
        i_binary_target = np.where((target == i), 1, 0)
        i_binary_predict = predict[:, i]
        roc_auc_per_label.append(roc_auc_score(i_binary_target, i_binary_predict))

    roc_auc_mean = np.mean(roc_auc_per_label).mean().item()

    return roc_auc_mean
