import numpy as np
from Layers.Initializers import *


class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size  = input_size
        self.output_size = output_size
        self.weights_w_prime = np.random.uniform(0.0,1.0,(self.input_size, self.output_size))
        self.bias_b_prime = np.random.uniform(0.0,1.0,(1, self.output_size))
        self.optimizer = None
        self.weights = np.vstack((self.weights_w_prime, self.bias_b_prime))

        self.input_tensor = None

    def forward(self, input_tensor):

        input_tensor1 = np.hstack((input_tensor, np.ones((input_tensor.shape[0],1))))
        self.input_tensor = input_tensor1
        output = np.dot(input_tensor1, self.weights)

        return output

    def backward(self, error_tensor):

        self.dWeights = np.dot(self.input_tensor.T, error_tensor)

        error_tensor_passed = np.dot(error_tensor, self.weights.T)

        if self.optimizer != None:
            self.weights = self.optimizer.calculate_update(self.weights, self.dWeights)


        return error_tensor_passed[:,0:-1]

    @property
    def gradient_weights(self):
        return self.dWeights

    @property
    def set_optimizer(self):
        return self.optimizer


    @set_optimizer.setter
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    # (M) WRONG!
    # def initialize(self, weights_initializer, bias_initializer):
        # self.weights_w_prime  = weights_initializer.initialize((self.input_size, self.output_size),self.input_size,self.output_size)
        # self.bias_b_prime = bias_initializer.initialize((self.input_size, self.output_size),self.input_size,self.output_size)

    # def initialize(self, a=None, b=None):
    #     pass


    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = self.weights[:-1, :].shape
        bias_shape = self.weights[-1, :].reshape(1,-1).shape
        self.weights[:-1, :] = weights_initializer.initialize(weights_shape, weights_shape[0], weights_shape[1])
        self.weights[-1, :] = bias_initializer.initialize(bias_shape, weights_shape[0], weights_shape[1])
        pass