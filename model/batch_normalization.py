import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x)
        gamma = np.array(gamma).reshape(1, -1)
        beta = np.array(beta).reshape(1, -1)
        running_mean = np.array(running_mean).reshape(1, -1)
        running_var = np.array(running_var).reshape(1, -1)
        if training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            x_norm = (x - mean)/np.sqrt(var + eps)
        else:
            x_norm = (x - running_mean)/np.sqrt(running_var + eps)
        y = np.round((x_norm * gamma) + beta, 4)
        y = y.tolist()
        if training:
            running_mean = np.round(momentum*mean + (1-momentum)*running_mean, 4)
            running_var = np.round(momentum*var + (1-momentum)*running_var, 4)
        running_mean = running_mean.reshape(-1).tolist()
        running_var = running_var.reshape(-1).tolist()
        return (y, running_mean, running_var)
