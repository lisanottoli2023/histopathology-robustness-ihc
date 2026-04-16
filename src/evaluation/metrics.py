from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



def compute_metrics(y_true, y_pred, class_names):
    """
    Compute precision, recall, F1-score, and accuracy for each class.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        class_names (list): List of class names.
    Returns:
        dict: A dictionary containing precision, recall, F1-score, and accuracy for each class.
    """
    precision, recall, f1_scores, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)), zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    return {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "per_class": {
        class_name: {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1_scores[i],
            "support": support[i]
        }
        for i, class_name in enumerate(class_names)
    }
}




