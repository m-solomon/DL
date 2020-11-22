import numpy as np


class Flatten:
    def __init__(self):
        pass

    def forward(self, input_tensor):

        self.input_tensor = np.asarray(input_tensor)
        s = self.input_tensor.shape
        returned = self.input_tensor.reshape((s[0] , s[1] * s[2] * s[3]))

        return returned


    def backward(self, error_tensor):
        shape = self.input_tensor.shape
        returned = error_tensor.reshape(shape)

        return returned

