import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def precision_recall_at_k(predictions, k=10):
    """
    predictions: dict user_id -> list of tuples (item_id, score, is_relevant_bool)
    returns (precision_at_k, recall_at_k) averaged over users with at least one relevant item
    """
    precisions = []
    recalls = []
    for u, items in predictions.items():
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        topk = items_sorted[:k]
        relevant_total = sum(1 for it in items if it[2])
        if relevant_total == 0:
            continue
        rel_in_topk = sum(1 for it in topk if it[2])
        precisions.append(rel_in_topk / k)
        recalls.append(rel_in_topk / relevant_total)
    if len(precisions) == 0:
        return 0.0, 0.0
    return float(np.mean(precisions)), float(np.mean(recalls))