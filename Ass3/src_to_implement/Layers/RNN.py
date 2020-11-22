import numpy as np
from Layers import Base
from Layers import FullyConnected
from Layers.TanH import *
from Layers.Sigmoid import *
from Layers.FullyConnected import *
from Layers.Initializers import *
import copy


class RNN(Base.BaseLayer): 
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(testing_phase = False)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.node1_input_size = self.hidden_size + self.input_size
        
        self.hidden_state = np.zeros(self.hidden_size)
        
        
        self.memorize = False
        self.optimizer= None

        
        self.FC_obj_1 = FullyConnected(self.node1_input_size, self.hidden_size)
        self.FC_obj_2 = FullyConnected(self.hidden_size, self.output_size)
        self.TanH_obj = TanH()
        self.Sig_obj = Sigmoid()
        
        
        
#######################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################    
  
    
    
    def forward(self, input_tensor):
        
        batch_size = input_tensor.shape[0]
        if self.memorize == False:
            self.hidden_state = np.zeros(self.hidden_size)

        self.Y = np.zeros((batch_size, self.output_size))
        self.FC1_cache = []
        self.FC2_cache = []
        self.Sig_cache = []
        self.TanH_cache = []
        
        for i in range(batch_size):
            
            node1_input = np.concatenate((self.hidden_state.reshape(self.hidden_size,1),
                                          input_tensor[i].reshape(input_tensor.shape[1],1)))  #concatenated input
            FC_1_f =  self.FC_obj_1.forward(node1_input.T)                          #first FC layer
            self.FC1_cache.append(self.FC_obj_1.input_tensor_prime_w_bias_factors)
            
            self.hidden_state = self.TanH_obj.forward(FC_1_f)                          #h_t
            self.TanH_cache.append(self.TanH_obj.tan_H)
            
            FC_2_f = self.FC_obj_2.forward(self.hidden_state)                     #second FC layer 
            self.FC2_cache.append(self.FC_obj_2.input_tensor_prime_w_bias_factors)
            
            
            self.Y[i] = self.Sig_obj.forward(FC_2_f) 
            self.Sig_cache.append(self.Sig_obj.sigomd)   
        
        
        return self.Y   
            
            
            
            

 
 #######################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
        
    def backward(self, error_tensor):
        Backward_out = np.zeros((error_tensor.shape[0], self.input_size))
        hidden_state_grad = np.zeros(self.hidden_size)
        
        self.grad_2 = 0.0
        self.grad_1 = 0.0
        batch_size = error_tensor.shape[0]
        for i in reversed(range (batch_size)):
            
            self.Sig_obj.sigomd = self.Sig_cache[i]
            Sig_out = self.Sig_obj.backward(error_tensor[i])
            
            self.FC_obj_2.input_tensor_prime_w_bias_factors = self.FC2_cache[i]
            FC_2_b = self.FC_obj_2.backward(Sig_out)
            self.grad_2 += (self.FC_obj_2.grad_weight)
            
            sumation_node2 = FC_2_b + hidden_state_grad
            
            self.TanH_obj.tan_H = self.TanH_cache[i]
            TanH_out = self.TanH_obj.backward(sumation_node2)
            
            self.FC_obj_1.input_tensor_prime_w_bias_factors = self.FC1_cache[i]
            FC_1_b = self.FC_obj_1.backward(TanH_out)
            self.grad_1 += (self.FC_obj_1.grad_weight)
            
            Backward_out[i] = np.squeeze(FC_1_b.T[self.hidden_size::])
            hidden_state_grad = np.squeeze(FC_1_b.T[0:self.hidden_size])
            
            
        #optimization
        self.grad_weight_2 = np.asarray(self.grad_2)  #outer
        self.grad_weight_1 = np.asarray(self.grad_1) #hidden

        if (self.optimizer != None):
            self.FC_obj_1.weights = self.optimizer.calculate_update(self.FC_obj_1.weights, self.grad_weight_1)
            self.FC_obj_2.weights = self.optimizer.calculate_update(self.FC_obj_2.weights, self.grad_weight_2)

        return Backward_out
        
        

            
          
            
            
            
        
        
        
        
    
########################################################################################################################

    def initialize(self, weights_initializer, bias_initializer):
        if (weights_initializer != None) and (bias_initializer != None):
            self.FC_obj_1.initialize(weights_initializer, bias_initializer)
            self.FC_obj_2.initialize(weights_initializer, bias_initializer)
        
    
        
        
 ###########################################################################################################################################################################################################


    @property
    def memorizes(self):
        return self.memorize

    @memorizes.setter
    def memorizes(self, var):
        self.memorize = var
        
        
    @property
    def weights(self):
        return self.FC_obj_1.return_w()

    @weights.setter
    def weights(self, w):
        self.FC_obj_1.weights = w


    @property
    def gradient_weights(self):
        return self.grad_1
    
    
    
    
    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = copy.deepcopy(optimizer)


#    @property
#    def optimizer2(self):
#        return self.__optimizer
    

#    @optimizer2.setter
#    def optimizer2(self, optimizer2):
#        self.__optimizer = copy.deepcopy(optimizer2)

    
   
