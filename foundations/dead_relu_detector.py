import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        with torch.no_grad():
            ans = []
            out = x
            for layer in model:
                out = layer(out)
                if isinstance(layer, nn.ReLU):
                    dead_frac_arr = (out==0).all(dim=0)
                    dead_frac = dead_frac_arr.float().mean().item()
                    ans.append(round(dead_frac, 4))
        return ans

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if max(dead_fractions) > 0.5:
            return 'use_leaky_relu'
        
        if dead_fractions[0] > 0.3:
            return 'reinitialize'
        
        arr = [0 if dead_fractions[i] < dead_fractions[i+1] else 1 for i in range(len(dead_fractions) - 1)]
        if sum(arr) == 0 and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'
        
        return 'healthy'
