import scipy.misc
import numpy as np



class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor
    
    
class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = (self.momentum_rate * self.v) - (self.learning_rate*gradient_tensor)
        return weight_tensor + self.v
    
    
    
    
class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.counter_k = 1
        
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = (self.mu * self.v) + ( (1-self.mu) * gradient_tensor )
        self.r = (self.rho * self.r) + ( (1-self.rho) * (gradient_tensor * gradient_tensor) )
        
        self.v_cap = (self.v) / (1 - self.mu**(self.counter_k)) 
        self.r_cap = (self.r) / (1 - self.rho**(self.counter_k))
        
        self.counter_k = self.counter_k + 1 
        
        ebsilon = np.finfo(float).eps
        
        return weight_tensor - self.learning_rate*( (self.v_cap) / (np.sqrt(self.r_cap)+ebsilon) )