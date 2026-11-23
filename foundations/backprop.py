import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        l = 1
        z = np.dot(x, w) + b
        y_pred = 1/(1 + np.exp(-z))
        dz = (y_pred - y_true) * y_pred * (1 - y_pred)
        dw = np.round((1/l) * dz * x, 5)
        db = np.round(dz/l, 5)
        return (dw, db)


