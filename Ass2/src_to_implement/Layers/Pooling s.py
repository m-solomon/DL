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

    def forward(self,input_tensor):

        self.input_tensor = input_tensor
        if len(input_tensor.shape) == 3: 
            self.input_tensor = input_tensor.reshape((input_tensor.shape + (1,)))

        
        self.in_batch_size =  input_tensor.shape[0]
        self.in_Nr_channels = input_tensor.shape[1]
        self.in_y =           input_tensor.shape[2]
        self.in_x =           input_tensor.shape[3]
        
        out_y = input_tensor.shape[2] - self.pooling_shape[0] + 1
        out_x = input_tensor.shape[3] - self.pooling_shape[1] + 1
        
        
        output_shape = (self.in_batch_size,
                        self.in_Nr_channels,
                        out_y,
                        out_x)

        

        out =[]
        out_index = []

        for b in range(self.in_batch_size):
            for c in range(self.in_Nr_channels):

                for i in range(out_y):
                    for j in range(out_x):
                        max = 0
                        index = []

                        for y in range(i, i + self.pooling_shape[0]):
                            for x in range(j, j + self.pooling_shape[1]):
                                if input_tensor[b, c, y, x] > max:
                                    max = input_tensor[b, c, y,x]
                                    index = [y,x]

                        out.append(max)
                        out_index.append(index)
        out_array = np.asarray(out)
        out_array = out_array.reshape(output_shape)


        strided = out_array [:, :, ::int(self.stride_shape[0]), ::int(self.stride_shape[1])]

    
        outIndex_shape = (  input_tensor.shape[0],
                            input_tensor.shape[1],
                            input_tensor.shape[2]-self.pooling_shape[0]+1,
                            input_tensor.shape[3]-self.pooling_shape[1]+1,
                            2
                          )
        
        
        
        passed = np.asarray(out_index).reshape(outIndex_shape)
        self.indices = passed [:, :, :: int(self.stride_shape[0]), ::int(self.stride_shape[1]),:]


        return strided



    def backward (self, error_tensor):

        errorPlaceHolder = np.zeros_like(self.input_tensor)

        #if self.stride_shape == self.pooling_shape:

        for b in range(self.input_tensor.shape[0]): #looping through the training examples

            for c in range(self.input_tensor.shape[1]):  #looping through the number of channels in the training examples
                counter = 0
                for i in range(self.indices.shape[2]):
                    for j in range(self.indices.shape[3]):  # looping through the passed indices

                        index1 = self.indices[b,c,i,j][0]
                        index2 = self.indices[b,c,i,j][1]

                        errorPlaceHolder[b,c,index1,index2] += error_tensor[b,c].flatten()[counter]
                        counter += 1

        return errorPlaceHolder