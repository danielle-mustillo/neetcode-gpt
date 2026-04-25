import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Dropout(p=0.2)
        self.layer4 = nn.Linear(512, 10)
        self.layer5 = nn.Sigmoid()
        pass

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        
        print(images.shape)
        # print(images)
        x = images
        for module  in self.children():
            x = module(x)
            print(module)

        return x
