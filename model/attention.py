import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
        

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        
        # 1.
        K = self.key(embedded)
        print("K", K)
        Q = self.query(embedded)
        print("Q", Q)
        V = self.value(embedded)
        print("V", V)

        # 2.
        context_length, attention_dim = K.size(dim=1), K.size(dim=2)
        print("context_length", context_length, attention_dim)
        attention_scores = Q @ K.mT
        attention_scores = attention_scores / attention_dim**0.5
        print("attention_scores", attention_scores)
        
        # 3. 
        # batch_size = embedded.size(dim=0)
        # print("batch_size", batch_size)
        causal_mask = torch.tril(torch.ones(context_length, context_length))
        print("causal_mask", causal_mask)
        attention_scores_masked = torch.where(causal_mask <= 0.0, float('-inf'), attention_scores)
        # causal_mask[causal_mask <= 0.0] = float('-inf')
        print("attention_scores_masked", attention_scores_masked)
        
        
        # 4. softmax
        attention = nn.functional.softmax(attention_scores_masked, dim=2)
        print("attention", attention)

        # 5. 
        return torch.round(attention @ V, decimals=4) 
