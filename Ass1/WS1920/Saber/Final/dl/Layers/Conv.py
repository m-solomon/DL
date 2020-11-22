import numpy as np
import scipy
from scipy import signal
from Layers.Initializers import *
from copy import deepcopy

class Conv:
    def __init__(self, stride_shape, convolution_shape , num_kernels):
        self.optimizer = None
        self.optimizer_weights = None
        self.optimizer_biases = None
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape # [c,m] or [c,m,n]
        self.num_kernels = num_kernels

        if len(self.convolution_shape) == 2:  # 1x1 or 1D kernal [c,m]
            self.c_shape = (self.convolution_shape[0],self.convolution_shape[1],1)
            self.convolution_shape = self.c_shape

        else:  # 2D [c,m,n]
            self.c_shape = self.convolution_shape


        self.weights_shape = (self.num_kernels, self.c_shape[0], self.c_shape[1],self.c_shape[2])
        self.weights = UniformRandom().initialize(self.weights_shape)
        self.bias = UniformRandom().initialize((self.num_kernels))

        if len(self.stride_shape) == 1:
            self.stride_shape = np.asarray(self.stride_shape)
            self.s1 = self.stride_shape[0]
            self.s2 = self.stride_shape[0]
        else:
            self.s1 = self.stride_shape[0]
            self.s2 = self.stride_shape[1]


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights_shape, self.c_shape[0] * self.c_shape[1] * self.c_shape[2] , self.num_kernels * self.c_shape[1]*self.c_shape[2])
        self.bias = bias_initializer.initialize(self.num_kernels, self.c_shape[0]*self.c_shape[1]*self.c_shape[2],self.num_kernels*self.c_shape[1]*self.c_shape[2])


    def forward(self,input_tensor): #input tensor shape (b,c,y) or (b,c,y,x)
        self.input_tensor= input_tensor

        if len(self.input_tensor.shape) == 3:  # 1D [b,c,y]
            self.input_tensor = np.expand_dims(self.input_tensor, 3)
        self.output_shape = (self.input_tensor.shape[0],) + (self.num_kernels,) + self.input_tensor.shape[2:4]

        stacked_convolved = list()

        for b in range(self.input_tensor.shape[0]):

            n = self.input_tensor[b][:, ::int(self.s1), ::int(self.s2)]
            output_shape = n.shape

            correlated = np.zeros((self.num_kernels, output_shape[1], output_shape[2]))

            # channels = np.zeros_like(correlated)
            #
            # for c in range (self.input_tensor.shape[1]):
            #     channels += scipy.ndimage.correlate(self.input_tensor[b,c], self.weights[i,c], None, 'constant')
            #


            for i in range(self.num_kernels):
                temp = scipy.ndimage.correlate(self.input_tensor[b], self.weights[i], None, 'constant')
                correlated[i] = temp[self.input_tensor.shape[1] // 2][::int(self.s1), ::int(self.s2)] + self.bias[i]

            stacked_convolved.append(correlated)

        stacked_convolved_npArray = np.asarray(stacked_convolved)

        if stacked_convolved_npArray.shape[3] == 1:  # come back to the old dimentions (from 2D to 1D)
            stacked_convolved_npArray = stacked_convolved_npArray[:,:,:,0]

        return stacked_convolved_npArray


##################################### backward #####################################
    #
    # def downsample(self, tensor):
    #     down = tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
    #     return (down)

    def upsample(self, tensor):
        up = np.zeros(self.output_shape)
        up[:, :, ::self.s1, ::self.s2] = tensor
        return (up)

    def backward(self, error_tensor):

        self.weights_temp = np.transpose(self.weights, (1, 0, 2, 3))
        if len(error_tensor.shape) == 3:
            error_tensor = np.expand_dims(error_tensor, -1)

        error_tensor = self.upsample(error_tensor)


        self.weights_temp = np.flip(self.weights_temp, axis=1)

        ouput_placeholder = np.zeros_like(self.input_tensor)
        for b in range(error_tensor.shape[0]):
            for i in range(self.weights_temp.shape[0]):
                temp = scipy.ndimage.convolve(error_tensor[b], self.weights_temp[i], None, 'constant')
                temp2 = temp[error_tensor.shape[1] // 2]
                ouput_placeholder[b,i] = temp2


        stacked_npArray = np.asarray(ouput_placeholder)

        if stacked_npArray.shape[3] == 1:
            stacked_npArray = stacked_npArray[:, :, :, 0]



        p1 = self.convolution_shape[1]
        p2 = self.convolution_shape[2]

        if p1 % 2 == 0:
            px1 = p1 // 2
            px2 = p1 // 2 -1
        else:
            px1 = p1 // 2
            px2 = p1 // 2

        if p2 % 2 == 0:
            py1 = p2 // 2
            py2 = p2 // 2-1
        else:
            py1 = p2 // 2
            py2 = p2 // 2

        paded = np.pad(self.input_tensor, ((0, 0),(0,0) ,(px1, px2), (py1, py2)), 'constant', constant_values=(0, 0))

        gradient_w = np.zeros_like(self.weights)

        for c in range(error_tensor.shape[1]):
            temp = np.zeros(self.weights.shape[1:])
            for b in range(error_tensor.shape[0]):
                temp += scipy.signal.correlate(paded[b] , np.expand_dims(error_tensor[b][c], 0), mode = "valid")
            gradient_w[c] = temp


        self.gradient_weights = gradient_w


        sum = np.zeros((error_tensor.shape[1],1)) # c x 1
        sums = np.zeros((error_tensor.shape[1],1)) # c x 1

        for b in range(error_tensor.shape[0]):
            for c in range(error_tensor.shape[1]):
                sum[c] = np.sum(error_tensor[b,c])
            sums += sum


        self.gradient_bias = sums


        if self.optimizer != None:

            self.weights = self.optimizer_weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer_biases.calculate_update(self.bias, self.gradient_bias.reshape(-1))


        return stacked_npArray



    @property
    def optimizer(self):
        return self.__optimizer


    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = deepcopy(optimizer)
        self.optimizer_weights = deepcopy(optimizer)
        self.optimizer_biases = deepcopy(optimizer)

