import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        rows = fan_out
        cols = fan_in
        
        std = math.sqrt(2 / (fan_in + fan_out))
        # print(std)
        out = torch.randn(fan_out, fan_in) * std
        # print(out)
        return out

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        return self.xavier(fan_in, fan_out).tolist()

    def kaiming(self, fan_in, fan_out):
        
        std = math.sqrt(2 / fan_in)
        # print(std)
        out = torch.randn(fan_out, fan_in) * std
        # print(out)
        return out

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        return self.kaiming(fan_in, fan_out).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        
        stddevs = []
        torch.manual_seed(0)
        x = None
        weights = []
        for layer in range(0, num_layers):
            
            if init_type == "kaiming":
                W = self.kaiming(input_dim, hidden_dim)
            elif init_type == "xavier":
                W = self.xavier(input_dim, hidden_dim)
            else:
                W = torch.randn(hidden_dim, input_dim)
            weights.append(W)
        
        
        
        x = torch.randn(1, input_dim)

        for W in weights:
            a = torch.relu(x @ W.T) 
            x = a
            out = round(a.std().item(), 2)
            print(out)
            stddevs.append(out)
        return stddevs
    
    # def relu(self, z):
    #     return np.maximum(0, z)
