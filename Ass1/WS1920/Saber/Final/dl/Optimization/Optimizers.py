import numpy as np
# from Optimization import *
# from Layers import *


class Sgd():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return  updated_weight_tensor


class SgdWithMomentum():
    def __init__(self, learning_rate, momentum_term):
        self.learning_rate = learning_rate
        self.momentum_term = momentum_term
        self.count = 0


    def calculate_update(self, weight_tensor, gradient_tensor):
        #self.weight_tensor = weight_tensor
        self.count = self.momentum_term * self.count - self.learning_rate * gradient_tensor
        weight_tensor = weight_tensor + self.count
        return weight_tensor

class Adam():
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.countv = 0
        self.countr = 0
        self.expcount = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        #self.weight_tensor = weight_tensor
        self.gradient_tensor = gradient_tensor
        # self.countv = self.countv / (1 - self.mu ** self.expcount)
        self.countv = self.mu * self.countv + (1 - self.mu) * self.gradient_tensor
        self.countvv = self.countv / (1-self.mu ** self.expcount)
        self.countr = self.rho * self.countr + (1 - self.rho) * (self.gradient_tensor * self.gradient_tensor)
        self.countrr = self.countr / (1 - self.rho ** self.expcount)

        # self.countr = self.countr / (1-self.rho ** self.expcount)

        self.expcount += 1

        # (M) WRONG!
        # weight_tensor = weight_tensor - self.learning_rate * (self.countvv + 0)/ (np.sqrt(self.countrr) + 0)

        epsilon = 1E-8
        weight_tensor = weight_tensor - self.learning_rate * (self.countvv + epsilon) / (np.sqrt(self.countrr) + epsilon)

        return weight_tensor