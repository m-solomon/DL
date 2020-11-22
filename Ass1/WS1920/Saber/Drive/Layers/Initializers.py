import numpy as np
class Constant:
    def __init__(self, constant_value = 0.1):
        self.constant_value = constant_value
    def initialize(self, weights_shape, fan_in = None, fan_out = None):
        initialized_tensor = self.constant_value * np.ones((weights_shape))
        return initialized_tensor

class UniformRandom:
    def __init__(self):
        pass
    def initialize(self, weights_shape, fan_in = None, fan_out = None ):
        # dims = (fan_out / (weights_shape[0] * weights_shape[1]), fan_in / (weights_shape[0] * weights_shape[1]),
        #         weights_shape[0], weights_shape[1])
         dims = weights_shape

         initialized_tensor = np.random.uniform(0, 1, dims)
         return initialized_tensor

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # dims = (fan_out / (weights_shape[0] * weights_shape[1]), fan_in / (weights_shape[0] * weights_shape[1]), weights_shape[0], weights_shape[1])
        dims = (weights_shape)
        std_dev = np.sqrt(2/(fan_out+fan_in))
        initialized_tensor = np.random.normal(0, std_dev, dims)
        return initialized_tensor

class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # dims = (fan_out / (weights_shape[0] * weights_shape[1]), fan_in / (weights_shape[0] * weights_shape[1]), weights_shape[0], weights_shape[1])
        std_dev = np.sqrt(2/(fan_in))
        dims = (weights_shape)
        initialized_tensor = np.random.normal(0, std_dev, dims)
        return initialized_tensor


