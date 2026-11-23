import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            l = len(data)
            indices = torch.randint(low=0, high= l - context_length, size=(batch_size, ))
            X = torch.stack([data[i:i + context_length] for i in indices])
            Y = torch.stack([data[i + 1:i + 1 + context_length] for i in indices])
            logits = model(X) # (batch_size, context_length, vocab_size)
            vocab_size = logits.shape[-1]
            logits_flat = torch.reshape(logits, (batch_size * context_length, vocab_size))
            Y_flat = torch.reshape(Y, (batch_size * context_length, ))
            loss = F.cross_entropy(logits_flat, Y_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return round(loss.item(), 4)



            


        

            

        
