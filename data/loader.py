import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:

        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        torch.manual_seed(0)
        X = torch.empty((batch_size, context_length), dtype=torch.long)
        Y = torch.empty((batch_size, context_length), dtype=torch.long)
        start_indices = torch.randint(low=0, high=(data.shape[0] - context_length), size=(batch_size, ))
        start = 0
        for i, val in enumerate(start_indices.flatten()):
            indice = val.item()
            X[i] = data[indice: indice + context_length]
            Y[i] = data[indice + 1: indice + 1 + context_length]
            start += context_length
        return (X, Y)





        
