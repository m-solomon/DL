import numpy as np
from Layers import Base

class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__(testing_phase = False) 
        self.probability = probability

    def forward(self,input_tensor):
        if self.testing_phase == False:
            self.mask = (np.random.random(input_tensor.shape) < self.probability)
            return self.mask * input_tensor * (1/self.probability)
        
        else:
            return input_tensor

    def backward(self,error_tensor):
        return error_tensor * self.mask / self.probability

