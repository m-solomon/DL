import numpy as np



######################################################## Constant ########################################################
class Constant:
    def __init__(self, constant = 0.1):
        self.constant = constant
        
    def initialize(self, weights_shape, fan_in= None, fan_out= None):  #ignore fan_in & fan_out
        return self.constant*(np.ones((weights_shape)))

########################################################### UniformRandom #####################################################    
    
class UniformRandom:
    def __init__(self):
        pass
    
    def initialize(self, weights_shape, fan_in= None, fan_out= None):  #ignore fan_in & fan_out
        return np.random.uniform(0, 1, weights_shape)

        
########################################################### Xavier/Glorot #####################################################        
class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_out+fan_in))
        return np.random.normal(0, sigma, weights_shape)

    
    
############################################################### He ######################################################
class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in))
        return np.random.normal(0, sigma, weights_shape)