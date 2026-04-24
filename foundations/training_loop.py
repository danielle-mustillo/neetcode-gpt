import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        y = y.reshape(-1, 1)
        n_samples, n_features = X.shape
        print(n_samples, n_features)
        w =  np.zeros((n_features, 1))
        b = np.zeros((n_samples, 1))
        print("w", w)
        for epoch in range(0, epochs):
            y_hat = X @ w + b
            # print("y_hat", y_hat)
            error = y_hat - y
            loss = np.mean(np.square(error))
            print("loss", loss)

            # print("X.T", X.T)
            # print("error", error)
            dW = 2 / len(y) * X.T @ error
            # dW = dW.reshape(-1)
            # print("dW", dW)
            db = 2 * np.mean(error)
            # db = db.reshape(-1)
            # print("db", db)

            w = w - lr * dW
            print("w", w)
            # w = w.flatten()
            b = b - lr * db
            # b = b.flatten()

        return (np.round(w.flatten(), 5), np.round(b.flatten()[0], 5))
