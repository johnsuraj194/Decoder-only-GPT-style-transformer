import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Run with torch.no_grad(). Round to 4 decimals.
        ans = []
        with torch.no_grad():
            output = x
            for layer in model:
                output = layer(output)
                if isinstance(layer, nn.Linear):
                    hashmap = {}
                    dead_fraction_arr = [1] * output.shape[1]
                    hashmap['mean'] = round(torch.mean(output).item(), 4)
                    hashmap['std'] = round(torch.std(output).item(), 4)
                    out = (output<=0).all(dim=0)
                    dead_fraction = out.float().mean().item()
                    hashmap['dead_fraction'] = round(dead_fraction, 4)
                    ans.append(hashmap)
            return ans

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        pred = model(x)
        loss = nn.MSELoss()(pred, y)
        loss.backward()
        ans = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                hashmap = {}
                g = layer.weight.grad
                hashmap['mean'] = round(g.mean().item(), 4)
                hashmap['std'] = round(g.std().item(), 4)
                hashmap['norm'] = round(g.norm().item(), 4)
                ans.append(hashmap)
        return ans

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for stat in activation_stats:
            if stat['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        for stat in gradient_stats:
            if stat['norm'] > 1000:
                return 'exploding_gradients'
        
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        
        for stat in activation_stats:
            if stat['std'] < 0.1:
                return 'vanishing_gradients'
            if stat['std'] > 10:
                return 'exploding_gradients'
        
        return 'healthy'

