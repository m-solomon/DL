import numpy as np
from Layers import Base



class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__(testing_phase = False)

    def forward(self, input_tensor):
        self.sigomd = 1 / (1 + np.exp(-1 * input_tensor))
        return self.sigomd

    def backward(self, error_tensor):
        return ( self.sigomd * (1 - self.sigomd) ) * error_tensor