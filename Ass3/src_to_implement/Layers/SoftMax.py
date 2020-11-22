import numpy as np


class SoftMax:
    def __init__(self):
        pass
        
        
    def forward(self, input_tensor):
        x_k_cap = input_tensor - np.max(input_tensor, axis=1)[:,None]
        numerator = np.exp(x_k_cap)
        denominator = np.sum(np.exp(x_k_cap), axis=1)[:,None]
        self.activation = numerator / denominator
        
        return self.activation
    
    
      
    def backward(self, error_tensor):
        grad = error_tensor * self.activation
        E_next = self.activation * ( error_tensor - np.sum(grad,axis=1)[:,None] )
        
        return E_next
    
        
 