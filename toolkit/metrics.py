import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_performance(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluates the performance of a models prediction using the accauracy.
    precision, recall, and f1 score.

    Args:
        y_true (np.ndarray): The actual classes.
        y_pred (np.ndarray): The model predictions.

    Returns:
        dict: a dictionary containing the accuracy, precision, recall, and f1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> None:
    """
    Plots the confusion matrix of a model predictions.

    Args:
        cm (np.ndarray): The confusion matrix.
        class_names (list): The class names.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm, cmap='viridis')
    plt.colorbar(cax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, ha="right")
    ax.set_yticklabels(class_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha='center', va='center',
                     color='red')
            
    plt.tight_layout()