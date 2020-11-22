import numpy as np
from Layers import Base

class Dropout(Base.Base):
    def __init__(self, probability):
        super().__init__(phase= "train")
        self.praobability = probability

    def forward(self,input_tensor):

        d = input_tensor
        if self.phase == "train":
            self.mask = (np.random.random(input_tensor.shape) < self.praobability)
            d = self.mask * input_tensor * 1/self.praobability
        return d

    def backward(self,error_tensor):
        return error_tensor * self.mask

