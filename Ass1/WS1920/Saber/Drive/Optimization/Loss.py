import numpy as np
# from Layers import *
# from Optimization import *

class CrossEntropyLoss:
    def __init__(self):
        pass


    def forward(self, input_tensor, label_tensor):
        self.input_tensor_cel = input_tensor
        loss = - np.sum(np.log(input_tensor + np.finfo(float).eps) * label_tensor)
        return loss


    def backward(self, label_tensor):
        E = - np.divide(label_tensor, self.input_tensor_cel)
        return E
