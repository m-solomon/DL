import numpy as np
import scipy
from scipy import signal
from Layers.Initializers import *
from copy import deepcopy
from Optimization import Optimizers
from Layers import Base


class Conv(Base.BaseLayer):
    
    ######################################################### Constructor #####################################################
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__(testing_phase = False) ################################ Ex3 ################################
        
        #stride_shape                   #can be a single value or a tuple.
        #convolution_shape              #For 1D the shape is [c, m], whereas for 2D the shape is [c, m, n],
        
        self.num_kernels = num_kernels  #integer value.(H)
        
        #-----------------------reshaping stride_shape & convolution_shape-----------------
        if len(stride_shape) == 1:
            self.stride_shape = (stride_shape + stride_shape)
        else:
            self.stride_shape = stride_shape
            
        if len(convolution_shape) == 2:
            self.convolution_shape = (convolution_shape[0], convolution_shape[1], 1)
        else:
            self.convolution_shape = convolution_shape
        
        
        self.ker_Nr_channels = self.convolution_shape[0] # Nr of channels per sample (S) (for intputs and kernals)
        self.ker_m =           self.convolution_shape[1] # Kernal size in y direction
        self.ker_n =           self.convolution_shape[2] # Kernal size in x direction
        
                           
        
        #------------inialize the weights and bias uniformally and seperatly--------------
        self.weights_shape_wo_bias = ( (self.num_kernels,) + self.convolution_shape)
        #shape = (num_kernels(H), ker_Nr_channels(S), ker_m, ker_n)
        
        self.weights = UniformRandom().initialize(self.weights_shape_wo_bias)
        self.bias = UniformRandom().initialize(self.num_kernels)
        
        
        #---------------inialize optimizers-------------------
        self.optimizer = None #learning rate
        self.optimizer_weights = None
        self.optimizer_biases = None
        
    
    
    
    ######################################################## Forward ###################################################################    
    def forward(self, input_tensor):
        
        #-----------------------------------------reshape input tensor-----------------------------------
        if len(input_tensor.shape) == 3:
            self.input_tensor = input_tensor.reshape((input_tensor.shape + (1,)))
        else:
            self.input_tensor = input_tensor
        
        self.in_batch_size =  self.input_tensor.shape[0]
        self.in_Nr_channels = self.input_tensor.shape[1]
        self.in_y =           self.input_tensor.shape[2]
        self.in_x =           self.input_tensor.shape[3]
        
        self.output_shape = (self.in_batch_size,
                             self.num_kernels,
                             self.in_y,
                             self.in_x
                            )
        #output shape = (in_batch_size, num_kernels(H) , in_y, in_x)
        middle_channel_idx = self.in_Nr_channels // 2 

        #--------------------------------------------cross correlate--------------------------------
        all_CC_output = []
        
        for image in range(self.in_batch_size):
            outputs_per_img = []
            for kernal in range(self.num_kernels):
                corr_out = scipy.ndimage.correlate( self.input_tensor[image], 
                                                    self.weights[kernal]       , None, 'constant')
                #Now shape looks like this (in_Nr_channels, y_in, x_in)

                corr_out_strided = corr_out[middle_channel_idx][::self.stride_shape[0], ::self.stride_shape[1]] + self.bias[kernal]
                #Now shape looks like this (y_in/strid , x_in/stride) and add bias

                outputs_per_img.append(corr_out_strided)
                #Now shape looks like this (nr_kernals, y_in/strid, x_in/stride)

            all_CC_output.append(np.asarray(outputs_per_img))
            #Now shape looks like this (nr_images(BtchSize), nr_kernals, y_in/strid, x_in/stride)
                
            
            
        #----------------------------------------return output----------------------------
        Forward_out_4d_array = np.asarray(all_CC_output)
        
        if Forward_out_4d_array.shape[-1] == 1:
            return Forward_out_4d_array[:, :, :, 0]
            
        else:
            return Forward_out_4d_array
        
    
       
        
    ########################################################## Backward #################################################################
    def backward(self, error_tensor):
        #--------------------------------------reshape error tensor---------------------------------
        if len(error_tensor.shape) == 3:
            self.error_tensor = error_tensor.reshape((error_tensor.shape + (1,)))
        else:
            self.error_tensor = error_tensor
        #err_tens_shape = (nr_images(BtchSize), num_kernels(H), y_in/strid, x_in/stride)
        
        
        
        Backward_out_shape = self.input_tensor.shape     #backward_output_shape is same as input_tensor_shape
        Backward_output = np.zeros(Backward_out_shape)     
        
        middle_channel_idx = self.num_kernels // 2 
        
        #--------------------------------------upsample error tensor-------------------------------
        up_sampled_err_tensor = np.zeros(self.output_shape)
        up_sampled_err_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = self.error_tensor
        #upsampled_err_tens_shape = (nr_images(BtchSize), num_kernels(H) , in_y, in_x)
        
        
        #--------------------------------------rearrange and flipp weights--------------------
        weights_rearng = np.transpose(self.weights, (1, 0, 2, 3))
        #now the shape is (ker_Nr_channels(S), num_kernels(H), ker_m, ker_n)
        weights_rearng_flipped = np.flip(weights_rearng, axis=1)
        #shape still the same but filters are flippd

        
        

        #----------------------------------------Convolv-------------------------------------
        for err in range(self.in_batch_size):
            for kernal in range(weights_rearng.shape[0]): #or range(self.ker_Nr_channels)
                conv_out = scipy.ndimage.convolve( up_sampled_err_tensor[err], 
                                                   weights_rearng_flipped[kernal],          None, 'constant')
                #Now shape looks like this ( num_kernels(H) , in_y, in_x)
                
                Backward_output[err][kernal] = conv_out[middle_channel_idx]
                
        
        
        
        #-------------------------------------gradient_weights-------------------------------------
        m = self.ker_m
        n = self.ker_n
        
        if m % 2 != 0:
            pad_y1 = m // 2
            pad_y2 = pad_y1
        else:
            pad_y1 = m // 2
            pad_y2 = pad_y1 -1
            
        if n % 2 != 0:
            pad_x1 = n // 2
            pad_x2 = pad_x1
        else:
            pad_x1 = n // 2
            pad_x2 = pad_x1 -1
        
        
        padded_input_tensor = np.pad(self.input_tensor,
                                     ( (0,0), (0,0), (pad_y1,pad_y2), (pad_x1,pad_x2) ),
                                     'constant',
                                     constant_values=(0, 0)
                                    )
        
        
        grad_weights = np.zeros(self.weights.shape)

        for kernal in range(self.num_kernels):           # or error_tensor.shape[1] or (H)
            new_w = np.zeros(self.weights.shape[1:])     #or (ker_Nr_channels(S), ker_m, ker_n) 
            for image in range(self.in_batch_size):      #or error_tensor.shape[0] or (b)
                new_w = new_w + scipy.signal.correlate(padded_input_tensor[image],
                                                       up_sampled_err_tensor[image][kernal][np.newaxis],
                                                       mode = "valid")
            grad_weights[kernal] = new_w


        self.gradient_w = grad_weights
        
        #-------------------------------------Gradient_Bias--------------------------------------

        sum = np.zeros(self.num_kernels) 
        grad_bias = np.zeros(self.num_kernels)

        for err in range(self.in_batch_size):         # or error_tensor.shape[0]  (nr_images(BtchSize))
            for kernal in range(self.num_kernels):    # or error_tensor.shape[1]  (H)
                sum[kernal] = np.sum(up_sampled_err_tensor[err][kernal])
            grad_bias = grad_bias + sum


        self.gradient_b = grad_bias
        
        
        #-------------------------------------update Optimizers--------------------------------------


        if self.optimizer != None:
            self.weights = self.optimizer_weights.calculate_update(self.weights, self.gradient_w)
            self.bias = self.optimizer_biases.calculate_update(self.bias, self.gradient_b.reshape(-1))
            
        
        
        #--------------------------------------return output-------------------------------------------
        if Backward_output.shape[-1] == 1:
            return Backward_output[:, :, :, 0]
                                                
        else:
            return Backward_output
        
        
        
        
        
       
    
        
   
    
    

    
    
    ######################################################## Properties ###################################################################
  
    
    @property
    def gradient_weights(self):
        return self.gradient_w
    
    @property
    def gradient_bias(self):
        return self.gradient_b
    
    
    
    @property
    def optimizer(self):
        return self.__optimizer
    
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = deepcopy(optimizer)
        self.optimizer_weights = deepcopy(optimizer)
        self.optimizer_biases = deepcopy(optimizer)
        
        
        
        
        
    ########################################################### Initializer ################################################################
    
    #weights_initializer & bias_initializer could be {Constant(), UniformRandom(), Xavier(), He()}
    def initialize(self, weights_initializer, bias_initializer):  
        self.weights = weights_initializer.initialize(self.weights_shape_wo_bias,
                                                      self.ker_Nr_channels * self.ker_m * self.ker_n, #or self.convolution_shape[0]*[1]*[2]
                                                      self.num_kernels * self.ker_m * self.ker_n )
        self.bias = bias_initializer.initialize(self.num_kernels,
                                                self.ker_Nr_channels * self.ker_m * self.ker_n,       #or self.convolution_shape[0]*[1]*[2]
                                                self.num_kernels * self.ker_m * self.ker_n )
        
        pass 
    
    
    