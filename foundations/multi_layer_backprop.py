import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        print("yooo mama")
        z1 = np.dot(x.reshape(1,-1), np.array(W1.T)) + b1
        print("z1", z1)
        a1 = self.relu(z1)
        print("a1", a1)
        z2 = np.dot(a1, W2.T) + b2
        print("z2", z2)
        y_hat = z2

        loss = 1 / len(y_hat) * np.sum(np.square(y_hat - y_true))
        print("loss", loss)


        print("derivatives ")
        dz2 = (1 / len(y_hat) * 2 * (y_hat - y_true)).reshape(-1)
        # dz2 = 
        # a1 = 
        print("dz2", dz2)
        print("a1", a1.reshape(1,-1))
        dW2 = np.dot(dz2, a1.reshape(1, -1)).reshape(1,-1)
        print("dW2", dW2)
        db2 = dz2
        print("dW2", db2)

        da1 = dz2 * W2
        print("da1", da1)

        def my_func(a, b):
            if a > b:
                return a - b
            else:
                return a + b
        dz1 = np.where(z1 > 0, da1, 0)
        # dz1 = np.maximum(0, da1)
        print("dz1", dz1)

        db1 = dz1.reshape(-1)
        print("db1", db1)
        dW1 = np.dot(dz1.T, x.reshape(1, -1)) 
        print("dW1", dW1)

        return {
            "loss": np.round(loss, 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(dz2, 4).tolist()
        }
        pass

    def relu(self, z):
        return np.maximum(0, z)