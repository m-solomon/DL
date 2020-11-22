import numpy as np
from Layers import Base

class Sigmoid(Base.Base):
    def __init__(self):
        super().__init__(phase="train")

    def forward(self, input_tensor):
        self.sgmd = 1 / (1 + np.exp(-1 * input_tensor))
        return self.sgmd

    def backward(self, error_tensor):
        if len(self.sgmd.shape) == 1:
            s = self.sgmd[:, np.newaxis]
            return (s * (1 - s)) * error_tensor
        else:
            return ( self.sgmd * (1 - self.sgmd) ) * error_tensor


    # @property
    # def sgmd(self):
    #     return self.sgmd
    #
    # @sgmd.setter
    # def optimizer1(self, var):
    #     self.sgmd = var

