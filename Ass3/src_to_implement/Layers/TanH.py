import numpy as np
from Layers import Base


class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__(testing_phase = False)

    def forward(self, input_tensor):
        self.tan_H = np.tanh(input_tensor)
        return self.tan_H

    def backward(self, error_tensor):
        return error_tensor * (1- np.square(self.tan_H))