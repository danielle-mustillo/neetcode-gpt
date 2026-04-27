import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        final_loss = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            start_pos = torch.randint(0, len(data)-context_length , (batch_size,))

            X = torch.stack([data[i: i + context_length] for i in start_pos])
            Y = torch.stack([data[j: j + context_length] for j in start_pos + 1])

            logits = model(X)

            B, T, C = logits.shape

            logits_flat = logits.reshape((B * T, C ))
            targets_flat = Y.reshape((B * T,))


            optimizer.zero_grad()
            
            loss = F.cross_entropy(logits_flat, targets_flat)
            final_loss = loss
            loss.backward()
            optimizer.step()

            print(data)
            print("loss", loss)
        
        return round(final_loss.item(), 4)
        
