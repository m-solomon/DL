import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass
        
        
    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        ebsilon = np.finfo(float).eps
        loss = -np.sum( label_tensor * np.log(input_tensor + ebsilon) )
        
        return loss
    
    
      
    def backward(self, label_tensor):
        E_n = -np.divide( label_tensor, self.input_tensor)
        
        return E_n
    
        
 
    
    