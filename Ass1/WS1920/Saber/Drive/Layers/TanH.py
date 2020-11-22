import numpy as np
from Layers import Base

class TanH(Base.Base):
    def __init__(self):
        super().__init__(phase="train")

    def forward(self, input_tensor):
            self.th = np.tanh(input_tensor)
            return self.th

    def backward(self, error_tensor):
        if len(self.th.shape) == 1:
            s = self.th[:, np.newaxis]
            return  (1 - np.square(s)) * error_tensor
        else:
            return error_tensor * (1- np.square(self.th))


    # @property
    # def TanH(self):
    #     return self.TanH
    #
    # @TanH.setter
    # def optimizer1(self, var):
    #     self.TanH = var
