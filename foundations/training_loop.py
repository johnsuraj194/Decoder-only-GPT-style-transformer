import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # return (np.round(w, 5), round(b, 5))
        y = y.reshape(-1, 1)
        m, f = X.shape
        w = np.zeros((1, f))
        b = 0
        for i in range(epochs):
            pred = np.matmul(X, w.T) + b
            w_grad = 2/m * np.matmul((pred - y).T, X)
            b_grad = 2/m * np.sum(pred - y)
            w = w - lr*w_grad
            b = b - lr*b_grad
        
        return (np.round(w.reshape(-1), 5), round(b, 5))

        
