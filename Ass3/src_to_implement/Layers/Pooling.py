import numpy as np



class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        if len(stride_shape) == 1:
            self.stride_shape = (stride_shape + stride_shape)
        else:
            self.stride_shape = stride_shape
            
        if len(pooling_shape) == 1:
            self.pooling_shape = (pooling_shape + pooling_shape)
        else:
            self.pooling_shape = pooling_shape
    

    ######################################################## Forward ###################################################################    

    def forward(self,input_tensor):
        
        #-----------------------------------------reshape input tensor-----------------------------------
        self.input_tensor = input_tensor
        if len(input_tensor.shape) == 3:
            self.input_tensor = input_tensor.reshape((input_tensor.shape + (1,)))
        else:
            self.input_tensor = input_tensor
        
        
        self.in_batch_size =  input_tensor.shape[0]
        self.in_Nr_channels = input_tensor.shape[1]
        self.in_y =           input_tensor.shape[2]
        self.in_x =           input_tensor.shape[3]
        
        out_y = input_tensor.shape[2] - self.pooling_shape[0] + 1
        out_x = input_tensor.shape[3] - self.pooling_shape[1] + 1
 
        #-----------------------------------------find maximas & their indesies-----------------------------------
        
        maximas = []
        max_idxies = []
        for image in range(self.in_batch_size):
            for channel in range(self.in_Nr_channels):
                for i in range(out_y):
                    for j in range(out_x):
                        max_pixel = -1
                        max_idx = []
                        
                        for y in range( i, i + self.pooling_shape[0]):
                            for x in range(j, j + self.pooling_shape[1]):
                                if self.input_tensor[image][channel][y][x] > max_pixel:
                                    max_pixel = self.input_tensor[image][channel][y][x]
                                    max_idx = np.array([y,x])
                        maximas.append(max_pixel)
                        max_idxies.append(max_idx)
                        

        output_max_shape   = (self.in_batch_size,
                              self.in_Nr_channels,
                              out_y,
                              out_x)
        
        output_max         = np.asarray(maximas).reshape(output_max_shape)
        #shape is (batch_size, nr_channels_per_img, out_y, out_x)

        output_max_strided = output_max [:, :, ::int(self.stride_shape[0]), ::int(self.stride_shape[1])]
        #shape is (batch_size, nr_channels_per_img, out_y/stride, out_x/stride)

        
        out_idx_shape        = (output_max_shape + (2,))
        out_idx              = np.array(max_idxies).reshape(out_idx_shape)
        self.out_idx_strided = out_idx [:, :, ::int(self.stride_shape[0]), ::int(self.stride_shape[1]),:]

        return output_max_strided



    
    ########################################################## Backward ##############################################################

    def backward (self, error_tensor):
        
        sum = np.zeros_like(self.input_tensor)
        #shape = (BatchSize, Nr_channels, in_y, in_x)
        
        for image in range(self.in_batch_size):                                    #for every training examples
            for channel in range(self.in_Nr_channels):                             #for every channels in every training examples
                err_idx = 0                                                        #for every error in error tensor
                for i in range(self.out_idx_strided.shape[2]):                     #for every y index of a max
                    for j in range(self.out_idx_strided.shape[3]):                 #for every x index of a max
                            y = self.out_idx_strided[image,channel,i,j][0]
                            x = self.out_idx_strided[image,channel,i,j][1]
                            sum[image,channel,y,x] = sum[image,channel,y,x] + error_tensor[image,channel].flatten()[err_idx]
                            err_idx = err_idx + 1
                            
        
        Backward_output = sum

        return Backward_output