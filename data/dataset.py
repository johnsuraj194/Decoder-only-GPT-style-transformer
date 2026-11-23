import torch
from typing import List, Tuple
import numpy as np

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)
        tokens = raw_dataset.split()
        l = len(tokens)
        X = []
        Y = []
        indices = torch.randint(low=0, high=l-context_length, size=(batch_size, ))
        for i, indice in enumerate(indices.flatten()):
            X.append(tokens[indice: indice + context_length])
            Y.append(tokens[indice  + 1: indice + 1 + context_length])
        return (X, Y)
         
