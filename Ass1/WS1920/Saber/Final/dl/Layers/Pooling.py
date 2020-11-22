import numpy as np

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self,input_tensor):

        self.input_tensor = input_tensor

        output_shape = (input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2]-self.pooling_shape[0]+1,input_tensor.shape[3]-self.pooling_shape[1]+1)

        if len(input_tensor.shape) == 3:  # 1D [b,c,y]
            input_tensor = np.expand_dims(input_tensor, 3)

        out =[]
        out_index = []

        for b in range(input_tensor.shape[0]):
            for c in range(input_tensor.shape[1]):

                for i in range(0,input_tensor.shape[2] - self.pooling_shape[0] +1):
                    for j in range(0,input_tensor.shape[3] - self.pooling_shape[1]+1):
                        max = -9999999
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

        indexs_arr = np.asarray(out_index)
        outIndex_shape = (input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2]-self.pooling_shape[0]+1,input_tensor.shape[3]-self.pooling_shape[1]+1,2)
        passed = indexs_arr.reshape(outIndex_shape)

        strided_out = passed [:, :, :: int(self.stride_shape[0]), ::int(self.stride_shape[1]),:]

        self.indices = strided_out


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