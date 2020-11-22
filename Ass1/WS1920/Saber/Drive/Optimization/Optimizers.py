import numpy as np

class Base_Optimizer:
    def __init__(self):
        self.regularizer = None
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Base_Optimizer):
    def __init__(self, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):

        if self.regularizer != None:
            norm_gradient = self.regularizer.calculate_gradient(weight_tensor)
            gradient_tensor = gradient_tensor + norm_gradient

        updated_weight_tensor = weight_tensor - self.learning_rate * gradient_tensor

        return  updated_weight_tensor


class SgdWithMomentum(Base_Optimizer):
    def __init__(self, learning_rate, momentum_term):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_term = momentum_term
        self.count = 0


    def calculate_update(self, weight_tensor, gradient_tensor):


        self.count = self.momentum_term * self.count - self.learning_rate * gradient_tensor
        u_weight_tensor = weight_tensor + self.count


        if( self.regularizer!=None ):
            regularization_term = self.regularizer.calculate_gradient( weight_tensor )
            u_weight_tensor -= self.learning_rate * regularization_term

        return u_weight_tensor

class Adam(Base_Optimizer):
    def __init__(self, learning_rate, mu=0.999, rho=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.countv = 0
        self.countr = 0
        self.expcount = 1

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.gradient_tensor = gradient_tensor


        self.countv = self.mu * self.countv + (1 - self.mu) * self.gradient_tensor
        self.countvv = self.countv / (1-self.mu ** self.expcount)
        self.countr = self.rho * self.countr + (1 - self.rho) * (self.gradient_tensor * self.gradient_tensor)
        self.countrr = self.countr / (1 - self.rho ** self.expcount)

        self.expcount += 1


        epsilon = 1E-8
        U_weight_tensor = weight_tensor - self.learning_rate * (self.countvv + epsilon) / (np.sqrt(self.countrr) + epsilon)

        if( self.regularizer!=None ):
            regularization_term = self.regularizer.calculate_gradient( weight_tensor )
            U_weight_tensor -= self.learning_rate * regularization_term

        return U_weight_tensor