import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # return np.round(your_answer, 5)
        mean = np.mean(x)
        std = np.std(x)
        ans = ((x - mean)/(np.sqrt(std**2 + 1e-5))* gamma) + beta
        return np.round(ans, 5)     
