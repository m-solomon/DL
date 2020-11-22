import scipy.misc
import numpy as np


################################################### EX3: Base_Optimizer #########################################################
class Base_Optimizer:
    def __init__(self):
        self.regularizer = None
    
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
        
#################################################################################################################################



############################################################ SGD #########################################################
class Sgd(Base_Optimizer):
    def __init__(self, learning_rate):
        super().__init__()                                        ##### Ex3 #####
        self.learning_rate = learning_rate
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        
        if self.regularizer != None:                              ##### Ex3 #####
            reg_term = self.regularizer.calculate_gradient(weight_tensor)
            gradient_tensor = gradient_tensor + reg_term
        
        return weight_tensor - self.learning_rate * gradient_tensor
    
    
    
#################################################### SGD with Momentum #########################################################
class SgdWithMomentum(Base_Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()                                        ##### Ex3 #####
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = (self.momentum_rate * self.v) - (self.learning_rate*gradient_tensor)
        updated_weight_tensor = weight_tensor + self.v
        
        if self.regularizer != None:                              ##### Ex3 #####
            reg_term = self.regularizer.calculate_gradient(weight_tensor)
            updated_weight_tensor = updated_weight_tensor - self.learning_rate * reg_term

        return updated_weight_tensor
    
    
    

    
############################################################# ADAM #########################################################    
class Adam(Base_Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()                                        ##### Ex3 #####
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
        
        updated_weight_tensor = weight_tensor - self.learning_rate*( (self.v_cap) / (np.sqrt(self.r_cap)+ebsilon) )
        
        
        if self.regularizer != None:                              ##### Ex3 #####
            reg_term = self.regularizer.calculate_gradient(weight_tensor)
            updated_weight_tensor = updated_weight_tensor - self.learning_rate * reg_term
        
        
        return updated_weight_tensor