import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # Do not alter the fixed code below — it ensures reproducible test output.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        for i in range(new_chars):
            l = context.shape[1]
            logits = model(context[:,-context_length:])
            logit_vector = logits[:,-1,:]
            probs = nn.functional.softmax(logit_vector, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1, generator=generator)
            context = torch.cat((context, new_token), dim=1)
            generator.set_state(initial_state)

        ans = ""
        for id_num in context[0, 1:]:
            ans += int_to_char[id_num.item()]
        return ans
        # Once your code passes the test, check out the Colab link to see your code generate new Drake lyrics!
