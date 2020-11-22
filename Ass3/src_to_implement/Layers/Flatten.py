import numpy as np
from Layers import Base


class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__(testing_phase = False) ################################ Ex3 ################################
        
        
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        flat_dim = np.prod(np.asarray(self.input_tensor.shape[1:]))
        return self.input_tensor.reshape(self.input_tensor.shape[0], flat_dim)
        



    def backward(self, error_tensor):
        error_tensor_shape = self.input_tensor.shape
        return error_tensor.reshape(error_tensor_shape)