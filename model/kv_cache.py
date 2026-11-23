import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_k is None:
            self.cache_k, self.cache_v = new_k, new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
        return (self.cache_k, self.cache_v)

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        model_dim = x.shape[-1]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if kv_cache is None:
            kv_cache = KVCache()
        full_k, full_v = kv_cache.update(k, v)
        raw_scores = (torch.matmul(q, torch.transpose(full_k, 1, 2)))/model_dim ** 0.5 # (batch, 1, seq_len)
        attention_scores = torch.softmax(raw_scores, dim=-1) # (batch, 1, seq_len)
        output = torch.matmul(attention_scores, full_v)
        output = torch.round(output, decimals=4)
        return output, kv_cache







