from typing import List, Dict

class Solution:
    def helper(self, num, vocab):
        k_min = min([len(t) for t in vocab])
        k_max = max([len(t) for t in vocab])
        tokens = []
        num = str(num)
        l = len(num)
        i = 0
        j = k_max
        while i < l:
            while j-i >= k_min and num[i:j] not in vocab:
                j -= 1
            if j - i < k_min:
                tokens.append(num[i:i+1])
                i = i + 1
                j = i + k_max
            else:
                tokens.append(num[i:j])
                i = j
                j = i + k_max
        return tokens
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        
        ans = [self.helper(num, vocab) for num in numbers]
        return ans


    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self.helper(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        total_words = len(text.split(" "))
        total_tokens = self.count_tokens(text, vocab)
        return round(total_tokens/total_words, 4)
