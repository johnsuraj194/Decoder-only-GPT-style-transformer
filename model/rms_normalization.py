import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        
        # Return result rounded to 4 decimal places as a list
        x = np.array(x).reshape(1, -1)
        n = x.shape[1]
        gamma = np.array(gamma).reshape(1, -1)
        rms = np.sqrt((np.sum(x**2)/n) + eps)
        x_norm = x/rms
        output = np.round(x_norm * gamma, 4)
        return output.reshape(-1).tolist()


