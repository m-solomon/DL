import numpy as np
from Layers import Base
from Layers import Helpers

class BatchNormalization(Base.Base): #don't forget inheritance
    def __init__(self, channels):
        super().__init__(phase="train")
        self.optimizer = None
        self.channels = channels
        self.weights = []
        

# **********************************************************************************************************************#
# **********************************************************************************************************************#
    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape

##############################################################################
################### initializing the 1st mean and variance ###################
##############################################################################
        if len(self.weights) == 0:

            self.weights, self.bias = BatchNormalization.initialize(self)

            if len(input_tensor.shape) == 4:
                self.input_shape = input_tensor.shape
                input_tensor_init = self.reformat(input_tensor)
            else:
                input_tensor_init = input_tensor
            initialMean = np.mean(input_tensor_init, axis=0)
            initialVariance = np.mean((input_tensor_init - initialMean) ** 2, axis=0)

            self.mue = initialMean
            self.sigma = initialVariance

##############################################################################
############################### train phase ##################################
##############################################################################
        flag = 0
        self.input_tensor = input_tensor

        if self.phase == "train":
            self.alpha = 0.8


            if len(self.input_tensor.shape) == 4:
                self.input_shape = self.input_tensor.shape
                self.input_tensor = self.reformat(input_tensor)
                flag = 1

            meanVec = np.mean(self.input_tensor, axis=0)
            varianceVec = np.mean((self.input_tensor - meanVec) ** 2, axis=0)
            self.normalized_X = (self.input_tensor - meanVec) * 1.0 / np.sqrt(varianceVec + np.finfo(float).eps)
            # self.weights, self.bias  = BatchNormalization.initialize(self)
            out = self.weights * self.normalized_X + self.bias

            if flag == 1:
                out = self.reformat(out)

            self.mueT = self.alpha * self.mue + (1 - self.alpha) * meanVec
            self.sigmaT = self.alpha * self.sigma + (1 - self.alpha) * varianceVec
            self.mue = self.mueT
            self.sigma = self.sigmaT
            return out
##############################################################################
############################### test phase ###################################
##############################################################################
        else:

            if len(self.input_tensor.shape) == 4:
                self.input_shape = self.input_tensor.shape
                input_tensor = self.reformat(self.input_tensor)
                flag = 1


            self.input_tensor = input_tensor
            meanVec = self.mue
            varianceVec = (self.sigma)
            self.normalized_X = (self.input_tensor - meanVec) * 1.0 / np.sqrt(varianceVec + np.finfo(float).eps)

            out = self.weights * self.normalized_X + self.bias


            if flag == 1:
                out = self.reformat(out)

            return out

#**********************************************************************************************************************#
#**********************************************************************************************************************#

    def backward(self, error_tensor):
        flag = 4

        self.error_tensor = error_tensor ###########


        if len(error_tensor.shape) == 4:
            self.input_shape = error_tensor.shape
            e = self.reformat(error_tensor)
            flag = 5
        else:
            e = error_tensor



        input_gradient1 = Helpers.compute_bn_gradients(e, self.input_tensor, self.weights, self.mue, self.sigma, np.finfo(float).eps)

        # Gradients w.r.t weights and biases:

        self.gradient_weights = np.sum(self.normalized_X * e, axis=0, keepdims= True)

        self.gradient_bias = np.sum(e ,axis=0, keepdims= True)

        # Updating weights and biases:

        if( self.optimizer != None ):
            updated_weights = self.optimizer.calculate_update( self.weights, self.gradient_weights )
            updated_bias = self.optimizer.calculate_update( self.bias, self.gradient_bias )
            self.weights = updated_weights
            self.bias = updated_bias

        input_gradient = input_gradient1

        if flag == 5:
            input_gradient = self.reformat(input_gradient1)

        return input_gradient




    def initialize(self): # to initialize beta and gammmmmma

        self.gamas = np.ones((1, self.channels))
        self.beta = np.zeros((1, self.channels))

        return self.gamas, self.beta

    def reformat(self, tensor):

        b = self.input_shape[0]
        h = self.input_shape[1]
        m = self.input_shape[2]
        n = self.input_shape[3]

        if len(tensor.shape) == 4:
            step = tensor.reshape(b, h, m * n)
            step1 = np.transpose(step, (0, 2, 1))
            step2 = step1.reshape( b * m * n, h)
            return step2

        else:

            step = tensor.reshape(b, m * n, h)
            step1 = np.transpose(step, (0, 2, 1))
            step2 = step1.reshape(b, h, m, n)

            return step2


    def set_optimizer(self, optimizer):
           self.optimizer = optimizer

    ###################################################################
    @property
    def gradient_weightss(self):
        return self.gradient_weights

    @gradient_weightss.setter
    def gradient_weightss(self, var):
        self.gradient_weights = var
        return self.gradient_weight



    @property
    def gradient_biass(self):
        return self.gradient_bias

    @gradient_biass.setter
    def gradient_biasss(self, var):
        self.gradient_bias = var
        return self.gradient_bias

