import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        # print(z)
        adj_z = z - np.max(z)
        # print(adj_z)
        sum_elements = np.exp(adj_z)
        # print(sum_elements)
        totsum = np.sum(sum_elements)
        # print(totsum)
        softmax = np.divide(sum_elements, totsum)
        # print(softmax)
        return np.round(softmax, 4) 
