import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for i, cls_true in enumerate(classes):
        for j, cls_pred in enumerate(classes):
            matrix[i, j] = np.sum((y_true == cls_true) & (y_pred == cls_pred))

    return matrix
