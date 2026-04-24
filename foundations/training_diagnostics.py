import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        # print(model)
        # print(x)
        stats = []
        activations = x
        # with torch.no_grad():
        for name, layer in model.named_modules():
            # print(activations)
            print("lay your hands on me", layer)
            if not isinstance(layer, nn.Sequential):
                activations = layer(activations)
            if isinstance(layer, nn.Linear):
                # print('layer', layer.gradient)
                # print("activations", activations)
                mean = activations.mean().item()
                # print("mean", mean)
                std = activations.std().item()
                # print("std", std)
                maxes = torch.max(activations, dim=0).values
                # print("maxes", maxes)
                dead_fraction = (maxes <= 0).sum().item() / maxes.numel()
                # print("dead_fraction", dead_fraction)
                stats.append({
                    "mean": mean,
                    "std": std,
                    "dead_fraction": dead_fraction
                })
        return stats
        pass

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        stats = []
        y_pred = model(x)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(y_pred, y)
        model.zero_grad()
        loss.backward()
        
        #model[0].weight.grad
        for name, layer in model.named_modules():
            # print(activations)
            print("lay your hands on me", layer)
            # if not isinstance(layer, nn.Sequential):
            #     activations = layer(activations)
            if isinstance(layer, nn.Linear):
                mean = layer.weight.grad.mean()
                print('mean', mean)
                std = layer.weight.grad.std()
                print('std', std)
                norm = torch.norm(layer.weight.grad)
                print('norm', norm)
                stats.append({

                    "mean": mean.item(),
                    "std": std.item(),
                    "norm": norm.item()
                })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for stat in activation_stats:
            if stat["dead_fraction"] > 0.5:
                return "dead_neurons"
        for stat in gradient_stats:
            if stat["norm"] > 1000:
                return "exploding_gradients"
        if gradient_stats[-1]['norm'] < 10**-5:
            return "vanishing_gradients"
        for stat in activation_stats:
            if stat["std"] < 0.1:
                return "vanishing_gradients"
            elif stat["std"] > 10.0:
                return 'exploding_gradients'
        return "healthy"

