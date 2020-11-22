import numpy as np
# from Layers import *
# from Optimization import *
from Optimization import Loss


class SoftMax:
    def __init__(self):
        pass

    def forward(self, input_tensor):
        input_tensor = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        Sum = np.sum(np.exp(input_tensor), axis=1, keepdims=True)
        input_tensor = np.exp(input_tensor) / Sum
        self.op = input_tensor

        return self.op


    def backward(self, error_tensor):
        output = self.op
        x = error_tensor * output

        E_passed = output * (error_tensor - np.sum(x,axis=1 , keepdims=True))

        return E_passed