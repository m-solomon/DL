import numpy as np
from Layers import Base
from Layers import Helpers

class BatchNormalization(Base.BaseLayer): 
    def __init__(self, channels):
        super().__init__(testing_phase = False)
        self.optimizer = None
        self.channels = channels
        self.weights = []
        self.eps = np.finfo(float).eps
        
    def initialize(self, weights_initializer = 0, bias_initializer = 0): 
        gamma = np.ones((1, self.channels))
        beta = np.zeros((1, self.channels))
        return gamma, beta
    
    
    
    def reformat(self, tensor):

        
        if len(tensor.shape) == 4:
            self.b, self.c, self.m, self.n = tensor.shape
            tensor_reshaped = tensor.reshape(self.b, self.c, self.m * self.n)
            tensor_transposed = np.transpose(tensor_reshaped, (0, 2, 1))
            return tensor_transposed.reshape( self.b * self.m * self.n, self.c)

        else:
            tensor_reshaped = tensor.reshape(self.b, self.m * self.n, self.c)
            tensor_transposed = np.transpose(tensor_reshaped, (0, 2, 1))
            return tensor_transposed.reshape(self.b, self.c, self.m, self.n)
    
############################################# Forward ###################################    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor

#------------------------------------------ mean and variance ------------------------------------
        if len( self.weights) == 0:
            self.weights, self.bias = BatchNormalization.initialize(self)

            if len(input_tensor.shape) == 4:
                input_tensor_reshaped = self.reformat(input_tensor)
            else:
                input_tensor_reshaped = input_tensor
            
            self.mue = np.mean(input_tensor_reshaped, axis=0)
            self.sigma = np.mean((input_tensor_reshaped - self.mue) ** 2, axis=0)

#------------------------------------------------------ training ---------------------------------
        reformat_used = False
        
        if self.testing_phase == False:
            alpha = 0.8

            if len(self.input_tensor.shape) == 4:
                self.input_tensor = self.reformat(input_tensor)
                reformat_used = True

            mue_B = np.mean(self.input_tensor, axis=0)
            sigma_B = np.mean((self.input_tensor - mue_B) ** 2, axis=0)
            self.X_normal = (self.input_tensor - mue_B) * 1.0/ np.sqrt(sigma_B + self.eps)
            forward_output = self.weights * self.X_normal + self.bias

            if reformat_used:
                forward_output = self.reformat(forward_output)
                
                
            #### Moving AVG:
            self.mueT = alpha * self.mue + (1 - alpha) * mue_B
            self.sigmaT = alpha * self.sigma + (1 - alpha) * sigma_B
            self.mue = self.mueT
            self.sigma = self.sigmaT
            
            return forward_output
        
#-------------------------------------------------------- testing --------------------------------------
        else:

            if len(self.input_tensor.shape) == 4:
                self.input_tensor = self.reformat(self.input_tensor)
                reformat_used = True


            mue_B = self.mue
            sigma_B = self.sigma
            self.X_normal = (self.input_tensor - mue_B) / np.sqrt(sigma_B + self.eps)

            forward_output = self.weights * self.X_normal + self.bias


            if reformat_used:
                forward_output = self.reformat(forward_output)

            return forward_output
 
 ######################################################### Backward ###################################################################
        
    def backward(self, error_tensor):
        reformat_used = False
        if len(error_tensor.shape) == 4:
            error_tensor_reshaped = self.reformat(error_tensor)
            reformat_used = True
        else:
            error_tensor_reshaped = error_tensor



        #grad w.r.t inputs
        input_gradient = Helpers.compute_bn_gradients(error_tensor_reshaped, self.input_tensor, self.weights, self.mue, self.sigma)

        #grad w.r.t weights and biases
        self.gradient_w = np.sum(self.X_normal * error_tensor_reshaped, axis=0, keepdims= True)
        self.gradient_b = np.sum(error_tensor_reshaped ,axis=0, keepdims= True)


        #weights & biases
        if( self.optimizer != None ):
            self.weights = self.optimizer.calculate_update( self.weights, self.gradient_w )
            self.bias = self.optimizer.calculate_update( self.bias, self.gradient_b )



        if reformat_used:
            input_gradient = self.reformat(input_gradient)

        return input_gradient
        
        
   


        
        
 ###########################################################################################################################################################################################################
    def set_optimizer(self, optimizer):
           self.optimizer = optimizer
            
            
    @property
    def gradient_weights(self):
        return self.gradient_w

    @gradient_weights.setter
    def gradient_weights(self, var):
        self.gradient_w = var
        return self.gradient_weight



    @property
    def gradient_bias(self):
        return self.gradient_b

    @gradient_bias.setter
    def gradient_bias(self, var):
        self.gradient_bias = var
        return self.gradient_b

