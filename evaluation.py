from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_answer_accuracy(predictions, ground_truths):
    correct = 0
    total = len(ground_truths)

    for pred, truth in zip(predictions, ground_truths):
        if pred.strip().lower() == truth.strip().lower():
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


def evaluate_retrieval(predictions, ground_truths):
    """
    Compute precision, recall, and F1-score for retrieved results.
    
    - predictions: List of lists containing retrieved document titles.
    - ground_truths: List of lists containing expected relevant document titles.
    """

    precision_scores, recall_scores, f1_scores = [], [], []

    for pred_docs, true_docs in zip(predictions, ground_truths):
        # Convert lists to sets for comparison
        pred_set = set(pred_docs)
        true_set = set(true_docs)

        # Compute Precision, Recall, F1
        tp = len(pred_set & true_set)  # True Positives
        fp = len(pred_set - true_set)  # False Positives
        fn = len(true_set - pred_set)  # False Negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        "Precision": sum(precision_scores) / len(precision_scores),
        "Recall": sum(recall_scores) / len(recall_scores),
        "F1 Score": sum(f1_scores) / len(f1_scores)
    }