import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        vocab = []
        for sentence in (positive + negative):
            vocab += sentence.split(' ')
        vocab.sort()
        
        lookup = {}
        rank = 1.0
        for word in vocab:
            if word not in lookup:
                lookup[word] = rank
                rank += 1.0
        # print("lookup", lookup)
        
        all_tokens = []
        for sentence in (positive + negative):

            tokens = list(map(lambda word: lookup[word], sentence.split(' ')))
            all_tokens.append(torch.tensor(tokens))
            # tokens = 
            # print(tokens)
        
        # print(all_tokens)
        tokens = torch.nn.utils.rnn.pad_sequence(all_tokens, padding_value=0, batch_first=True)
        # print(tokens)
        return tokens
        # print(pos)
        # neg = torch.tensor(list(map(lambda word: lookup[word], negative)))
        # print(neg)

        pass
