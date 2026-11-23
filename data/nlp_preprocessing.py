import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        positive_tokens = [sentence.split() for sentence in positive]
        negative_tokens = [sentence.split() for sentence in negative]
        all_tokens = positive_tokens + negative_tokens
        vocab = [token for sentence in all_tokens for token in sentence]
        vocab = sorted(set(vocab))
        hashmap = {token : i + 1 for i, token in enumerate(vocab)}
        T = max([len(sentence) for sentence in all_tokens])
        n = len(positive_tokens)
        output = torch.zeros(2*n, T)
        for i in range(2*n):
            for j in range(len(all_tokens[i])):
                output[i][j] = hashmap[all_tokens[i][j]]
        return output




