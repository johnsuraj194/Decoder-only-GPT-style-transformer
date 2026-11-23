import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        #
        
        # Round to 5 decimal places. 

        matrix = np.array([[pos/(10000**((2*i)/d_model)) for i in range(d_model//2)] for pos in range(seq_len)])
        ans = np.zeros((seq_len, d_model))
        ans[:, 0::2] = np.sin(matrix)
        ans[:, 1::2] = np.cos(matrix)
        return np.round(ans, 5)
        