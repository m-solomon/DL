import numpy as np
import scipy
from scipy import signal
from Layers.Initializers import UniformRandom
from copy import deepcopy



class Conv:
    
    ###########################################################################################################################
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        #self.stride_shape = stride_shape                 #can be a single value or a tuple.
        #self.convolution_shape = convolution_shape       #For 1D the shape is [c, m], whereas for 2D the shape is [c, m, n],
                
        if len(stride_shape) == 1:
            self.stride_shape = (stride_shape + stride_shape)
        else:
            self.stride_shape = stride_shape
            
        if len(convolution_shape) == 2:
            self.convolution_shape = (convolution_shape[0], convolution_shape[1], 1)
        else:
            self.convolution_shape = convolution_shape
        
            
        self.ker_Nr_channels = self.convolution_shape[0] # should be equal to self.in_Nr_channels
        self.ker_m = self.convolution_shape[1]
        self.ker_n = self.convolution_shape[2]
        
        self.num_kernels = num_kernels                   #integer value.
        
        #######inialize the weights and bias uniformally and seperatly
        self.weights_shape_wo_bias = ( (self.num_kernels,) + self.convolution_shape)
        #shape = (num_kernels(H), ker_Nr_channels, ker_m, ker_n)
        self.weights = UniformRandom().initialize(self.weights_shape_wo_bias)
        self.bias = UniformRandom().initialize(self.num_kernels)
        
        
        self.optimizer = None #learning rate
        #self.optimizer_weights = None
        #self.optimizer_biases = None
        
    ###########################################################################################################################    
    def forward(self, input_tensor):
        
        if len(input_tensor.shape) == 3:
            self.input_tensor = input_tensor.reshape((input_tensor.shape + (1,)))
        else:
            self.input_tensor = input_tensor
        
        self.in_batch_size = self.input_tensor.shape[0]
        self.in_Nr_channels = self.input_tensor.shape[1]
        self.in_y = self.input_tensor.shape[2]
        self.in_x = self.input_tensor.shape[3]
        
        self.out_shape = 
        #output shape = (in_batch_size, num_kernels(H) , in_y, in_x)

        
        all_CC_output = []
        
        for image in range(self.in_batch_size):
            outputs_per_img = []
            for channel in range(self.in_Nr_channels):
                CC_maps_per_ch = []
                for kernal in range(self.num_kernels):
                    corr_out = scipy.ndimage.correlate(self.input_tensor[image][channel], self.weights[kernal][channel], None, 'constant')
                    #Now shape looks like this (y_in, x_in)
                    
                    corr_out_strided = corr_out[::self.stride_shape[0], ::self.stride_shape[1]] + self.bias[kernal] #downsamplin
                    #Now shape looks like this (y_in/strid , x_in/stride) and add bias
                    
                    CC_maps_per_ch.append(corr_out_strided)
                    #Now shape looks like this (nr_kernals, y_in/strid, x_in/stride)
                
                outputs_per_img.append(np.asarray(CC_maps_per_ch))
                #Now shape looks like this (nr_channels ,nr_kernals, y_in/strid, x_in/stride)
                
            all_CC_output.append(np.asarray(outputs_per_img))
            #Now shape looks like this (nr_images(BtchSize), nr_channels ,nr_kernals, y_in/strid, x_in/stride)
            
        
        Forward_out_5d_array = np.asarray(all_CC_output)
        
        if Forward_out_5d_array.shape[-1] == 1:
            return Forward_out_5d_array.reshape((self.in_batch_size,
                                                 self.in_Nr_channels,
                                                 self.num_kernels,
                                                 self.in_y//self.stride_shape[0]))
        else:
            return Forward_out_5d_array
        
    
    ###########################################################################################################################
    @property
    def gradient_weights(self):
        return #self.grad_weight
    
    @property
    def gradient_bias(self):
        return #self.grad_weight
    
    @property
    def optimizer(self):
        return self.optimizer
    
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self.optimizer = np.copy(optimizer)
        self.optimizer_weights = np.copy(optimizer)
        self.optimizer_biases = np.copy(optimizer)
    
        
    ###########################################################################################################################
    def backward(self, error_tensor):
        
        if len(error_tensor.shape) == 3:
            self.error_tensor = error_tensor.reshape((error_tensor.shape + (1,)))
        else:
            self.error_tensor = error_tensor
        
        up_sampled_err_tensor = np.zeros(self.output_shape)
            
        self.weights_flipped = np.transpose(self.weights, (1, 0, 2, 3))
        #now the shape is (nr_channels_ker, BtchSize, ker_m, ker_n)

        
        self.out_shape = 

        
        all_CC_output = []
        
        for image in range(self.in_batch_size):
            outputs_per_img = []
            for channel in range(self.in_Nr_channels):
                CC_maps_per_ch = []
                for kernal in range(self.num_kernels):
                    corr_out = scipy.ndimage.correlate(self.input_tensor[image][channel], self.weights[kernal][channel], None, 'constant')
                    #Now shape looks like this (y_in, x_in)
                    
                    corr_out_strided = corr_out[::self.stride_shape[0], ::self.stride_shape[1]] + self.bias[kernal] #downsamplin
                    #Now shape looks like this (y_in/strid , x_in/stride) and add bias
                    
                    CC_maps_per_ch.append(corr_out_strided)
                    #Now shape looks like this (nr_kernals, y_in/strid, x_in/stride)
                
                outputs_per_img.append(np.asarray(CC_maps_per_ch))
                #Now shape looks like this (nr_channels ,nr_kernals, y_in/strid, x_in/stride)
                
            all_CC_output.append(np.asarray(outputs_per_img))
            #Now shape looks like this (nr_images(BtchSize), nr_channels ,nr_kernals, y_in/strid, x_in/stride)
            
        
        Forward_out_5d_array = np.asarray(all_CC_output)
        
        if Forward_out_5d_array.shape[-1] == 1:
            return Forward_out_5d_array.reshape((self.in_batch_size,
                                                 self.in_Nr_channels,
                                                 self.num_kernels,
                                                 self.in_y//self.stride_shape[0]))
        else:
            return Forward_out_5d_array
        
        
        
        
        
        error_tensor_prim = error_tensor
        error_tensor_next = np.dot(error_tensor_prim, self.weights.T)
        self.grad_weight = np.dot(self.input_tensor_prime_w_bias_factors.T, error_tensor_prim)
        
        if self.optimizer != None:
            self.weights = self.optimizer.calculate_update(self.weights, self.grad_weight)
        
        return error_tensor_next[:,0:-1]
    
        
   
    
    ###########################################################################################################################
    def initialize(self, weights_initializer, bias_initializer): #weights_initializer & bias_initializer could be {Constant(), UniformRandom(), Xavier(), He()}
        self.weights[:-1, :] = weights_initializer.initialize(self.weights_prim_size, self.input_size, self.output_size)
        self.weights[-1, :] = bias_initializer.initialize(self.output_size, self.input_size, self.output_size)
                
        pass 
    