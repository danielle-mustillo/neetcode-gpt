import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        a = x#.reshape(1, -1)
        for i in range(0, len(weights)):
            
            weight = weights[i]
            bias = biases[i]
            print("a", a)
            print("weight", weight)
            print("weight.T", weight.T)
            a = np.dot(a, weight) + bias
            print("out", a)
            if i < len(weights) - 1:
                a = self.relu(a)
        return a
    
    def relu(self, x: NDArray[np.float64]):
        return np.maximum(0, x)