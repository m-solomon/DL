import numpy as np
from Optimization import Optimizers

class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights_size = (self.input_size, self.output_size )
        self.weights_wo_bias = np.random.uniform(0,1, self.weights_size) #without bias
        self.bias = np.random.uniform(0,1, self.output_size)   #self.bias = np.random.uniform(0,1, (1, self.output_size))
        self.weights =np.concatenate((self.weights_wo_bias, self.bias[None,:]),  axis=0)
        
        self.optimizer = None #learning rate
        
        
    def forward(self, input_tensor):
        input_tensor_prime = input_tensor
        batch_size = input_tensor_prime.shape[0]
        bias_factors = np.ones(batch_size, dtype='int') #np.ones(batch_size) #
        self.input_tensor_prime_w_bias_factors = np.concatenate((input_tensor_prime, bias_factors[:,None]),  axis=1) #transposed inputs with ones as factors for bias
        
        return np.dot(self.input_tensor_prime_w_bias_factors , self.weights)
    
    
    @property
    def set_optimizer(self):
        return self.optimizer
    @set_optimizer.setter
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    
    def backward(self, error_tensor):
        error_tensor_prim = error_tensor
        error_tensor_next = np.dot(error_tensor_prim, self.weights.T)
        self.grad_weight = np.dot(self.input_tensor_prime_w_bias_factors.T, error_tensor_prim)
        
        if self.optimizer != None:
            self.weights = self.optimizer.calculate_update(self.weights, self.grad_weight)
        
        return error_tensor_next[:,0:-1]
    
        
    @property
    def gradient_weights(self):
        return self.grad_weight
    
    
    
    def initialize(self, weights_initializer, bias_initializer): #weights_initializer & bias_initializer could be {Constant(), UniformRandom(), Xavier(), He()}
        self.weights[-1, :] = bias_initializer.initialize(self.output_size, self.input_size, self.output_size)
        self.weights[:-1, :] = weights_initializer.initialize(self.weights_size, self.input_size, self.output_size)
        
        
        
                
        pass 
    