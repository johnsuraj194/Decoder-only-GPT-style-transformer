import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embedding_layer = nn.Embedding(vocabulary_size, 16)
        self.linear_layer = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:

        # Return a B, 1 tensor and round to 4 decimal places
        x = x.to(torch.int32) 
        x = self.embedding_layer(x)
        x = x.mean(dim=1)
        x = self.linear_layer(x)
        x = self.sigmoid(x)
        return torch.round(x, decimals=4)
