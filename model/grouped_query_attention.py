import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape
        # self.q_proj(x) shape = (batch_size, T, num_heads*self.head_dim)
        Q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        #  Q is of shape (B, num_heads, T, head_dim)
        repeat = int(self.num_heads/self.num_kv_heads)
        K = K.repeat_interleave(repeat, dim=1)
        V = V.repeat_interleave(repeat, dim=1)
        raw_scores = torch.matmul(Q, K.transpose(2, 3))
        raw_scores = raw_scores/self.head_dim**0.5
        # raw_scores is of shape (B, num_heads, T, T)
        mask = torch.ones(T, T)
        mask = torch.tril(mask, diagonal=0) == 0
        raw_scores = raw_scores.masked_fill(mask, float('-inf'))
        attention_scores = torch.softmax(raw_scores, dim=-1)
        ans = torch.matmul(attention_scores, V).transpose(1, 2).reshape(B, T, self.num_heads*self.head_dim)
        return torch.round(self.output_proj(ans), decimals=4)
        # 6. Return rounded output (decimals=4)
        pass
