import numpy as np

class ReLU:
    def __init__(self):
        pass
        
        
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_tensor_next = np.copy(input_tensor)
        input_tensor_next[input_tensor_next <= 0] = 0
        
        return input_tensor_next
    
    
      
    def backward(self, error_tensor):
        error_tensor_next = (self.input_tensor >= 0) * error_tensor
        
        return error_tensor_next
    
        
 