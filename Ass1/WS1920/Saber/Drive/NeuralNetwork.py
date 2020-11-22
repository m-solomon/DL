import numpy as np
from copy import deepcopy
from Layers import Base
class NeuralNetwork(Base.Base):
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        super().__init__(phase="train")
        # self.phase = None ##########################################
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    @property # property decorators allow you to access class methods as object attributes
    def phase(self):
        return self.phase

    @phase.setter
    def phase(self, phas):
        self.__phase = phas


    def forward(self):
        data, label = self.data_layer.forward()
        self.label = label
        self.data = data


        temp = data
        for layer in self.layers:
            ###############################################################
            #layer.phase = self.phase ############ check this ##############  ## check ##
            ###############################################################
            temp = layer.forward(temp)

        loss = self.loss_layer.forward(temp, label)


        return loss


    def backward(self):
        temp = self.loss_layer.backward(self.label)

        for layer in reversed(self.layers):
            ###############################################################
            #layer.phase = self.phase ############ check this ##############
            ###############################################################
            temp = layer.backward(temp)




    def train(self, iterations):
        self.phase ="train"

        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()


    def test(self, input_tensor):
        self.phase = "test"
        temp = input_tensor
        for layer in self.layers:
            temp = layer.forward(temp)

        return temp


    def append_trainable_layer(self, layer):
        layer.optimizer = deepcopy(self.optimizer)
        # (M) WRONG!
        # layer.weights_initializer = deepcopy(self.weights_initializer)
        # layer.bias_initializer = deepcopy(self.bias_initializer)

        layer.initialize(deepcopy(self.weights_initializer), deepcopy(self.bias_initializer))
        self.layers.append(layer)