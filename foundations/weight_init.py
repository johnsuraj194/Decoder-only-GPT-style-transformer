import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()


    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        weights = []
        temp_input_dim = input_dim
        for layer in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (temp_input_dim + hidden_dim))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / temp_input_dim)
            else:
                std = 1.0
            weights.append(torch.randn(hidden_dim, input_dim) * std)
            temp_input_dim = hidden_dim
        
        stds = []
        x = torch.randn(1, input_dim)

        for layer in range(num_layers):
            z = x @ weights[layer].T
            a = torch.relu(z)
            std = round(a.std().item(), 2)
            stds.append(std)
            x = a
        return stds


        
