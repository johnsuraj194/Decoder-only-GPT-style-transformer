from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = [char for char in corpus]
        ans = []
        for num in range(num_merges):
            hashmap = {}
            max_count = 0
            for i in range(len(tokens) - 1):
                key = (tokens[i], tokens[i + 1])
                hashmap[key] = hashmap.get(key, 0) + 1
                max_count = max(hashmap[key], max_count)
            
            candidates = []
            for key, value in hashmap.items():
                if hashmap[key] == max_count:
                    candidates.append([key[0] + key[1], key])
            
            candidates.sort(key=lambda x: (x[1][0], x[1][1]))
            candidate = candidates[0]
            ans.append(list(candidate[1]))

            new_tokens = []
            merged = candidate[0]
            i = 0
            while i < (len(tokens) - 1):
                if tokens[i] + tokens[i + 1] == merged:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return ans





            
