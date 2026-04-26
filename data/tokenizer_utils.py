from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        numbers = [str(num) for num in numbers]
        tokens = []
        for num in numbers:
            tokens.append(self.tokenize(num, vocab))
            
        return tokens

    def tokenize(self, text: str, vocab):
        done = False
        tokens = []
        while not done:
            for token, id in sorted(vocab.items(), reverse=True):
                if text.startswith(token):
                    text = text[len(token):]
                    tokens.append(token)
                    done = len(text) <= 0
                    break;
        if len(text) > 0:
            raise ValueError("Could not find token which matches start string: [" + text + "]. Vocab: [" + str(vocab) + "]") # invalid state
        return tokens

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self.tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split(" ")
        
        count = self.count_tokens(text, vocab)

        return round(count / len(words), 4) 