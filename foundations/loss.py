import numpy as np
from numpy.typing import NDArray

'''
Binary (2): 
L = -1/n * sum(y * ln(pi) + (1-yi) * ln(1-pi))
problem: when pi == 0, log(0) => if 0, add epsilon

Categorical (3+):
L = -1/n * sum over n * (sum over Category yi log(pi))
'''

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7 
        y_pred[y_pred==0] = epsilon # y pred in [0,1]
        loss = -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
        return round(loss, 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred[y_pred==0] = epsilon
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return round(loss, 4)
