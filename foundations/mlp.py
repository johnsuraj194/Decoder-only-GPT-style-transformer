import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)

        num_layers = len(weights)
        input_layer = x
        for layer in range(num_layers):
            W = weights[layer]
            b = biases[layer]
            z = np.matmul(input_layer, W) + b
            if layer != num_layers - 1:
                a = np.maximum(z, 0)
            else:
                a = z
            input_layer = a
        
        return np.round(a, 5)
        
