import numpy as np


class L2_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        self.grad = self.alpha * weights
        return self.grad

#
    def norm(self, weights):
        self.weights = weights
        self.norm2 = np.linalg.norm(np.ravel(self.weights), ord = None) * self.alpha #should we divide by 2m?
        return self.norm2

class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        self.grad =  self.alpha * np.sign(weights)
        return self.grad

    def norm(self, weights):
        self.norm1 = np.linalg.norm(np.ravel(weights), 1) * self.alpha #should we divide by m?
        return self.norm1