import numpy as np
from copy import deepcopy
from Layers import Base


#
class NeuralNetwork(Base.BaseLayer):
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        super().__init__(testing_phase = False) ################################ Ex3 ################################
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        #self.testing_phase = False
        



    @property
    def phase(self):
        if self.layers:
            return self.layers[0].testing_phase
        pass
    
    @phase.setter
    def phase(self, ph):
        if ph:
            for layer in self.layers:
                layer.testing_phase = ph




    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.forward()

        #input_tensor_buffer = self.input_tensor
        input_tensor_buffer = np.copy(self.input_tensor)
        for L in self.layers:
            input_tensor_buffer = L.forward(input_tensor_buffer)

        return self.loss_layer.forward(input_tensor_buffer, self.label_tensor) #loss for 1 iteration




    def backward(self):
        buffer = np.copy(self.loss_layer.backward(self.label_tensor))

        for L in self.layers[::-1] :
            buffer = L.backward(buffer)
        
        pass 

    
    """def append_trainable_layer(self, layer):
        layer.optimizer = deepcopy(self.optimizer)
        self.layers.append(layer)"""
    
    """################################################################"""
    
    def append_trainable_layer(self, layer):
        layer.optimizer = deepcopy(self.optimizer)
        layer.initialize( deepcopy(self.weights_initializer), deepcopy(self.bias_initializer))
        #layer.initialize()
        self.layers.append(layer)
        
        pass 
    """################################################################"""



    def train(self, iterations):
        self.phase = False
        for i in range(iterations):
            
            self.loss.append(self.forward())
            self.backward()
        
        pass 




    def test(self, input_tensor):
        self.phase = True
        test = np.copy(input_tensor)
        for L in self.layers:
            
            test = L.forward(test)

        return test



