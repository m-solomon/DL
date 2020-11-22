import os.path
import json
import scipy
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt


# a = np.array([1,2,3,4])
#
#
# print(a)
#
# print(a)
# print(a.shape)
#
# b = np.array([[1,2,3,4]])
# print(b)
# print(b.shape)
#
# print(a.T.shape)
# print(b.T.shape)
#
# print('Finished')
#
# pass

#
# path = "exercise_data/0.npy"
# image = np.load(path)
#
# test = np.zeros([3, 4])
# x, y = np.ogrid[0:3, 0: 4]
#
#
# test[0:3, 0:4][x, y] = 1
# a =np.array([[1,2,3],
#             [3,4,5]])
#
# b =np.array([[1,2,3],
#             [1,2,3]])
# y = a*b
# x = np.sum( y ,axis=1 , keepdims=True)
# print(x.shape)
#
# print(a)
# print()
# print(b)
# print()
#
# print(y)
# print()
# f = x.reshape(x.shape[1],x.shape[0])
# print(x)
#
#
# def padwithzeros(vector, pad_width, iaxis, kwargs):
#      vector[:pad_width[0]] = 0
#      vector[-pad_width[1]:] = 0
#      return vector
#
# a = np.ones((3,2))
# print(a)
#
#
# print (np.lib.pad(a, ((4,3),(2,1)) , padwithzeros))

# kernals = []
#
# for i in range(3):
#     kernals.append([1,2])
#     kernals.append([[1,2,3],[1,2,3]])
#
#
# arr = np.ones((4,5,2,2))
#
# arr = np.pad(arr,((0,0),(0,0),(2,3),(1,1)), 'constant', constant_values = (0,0))
# print(arr.shape)

###############################


# input_tensor = np.array(range(np.prod((4, 4)) * 1), dtype=np.float)
# print(input_tensor)
#
# input_tensor = input_tensor.reshape((4, 4))
# print(input_tensor)

#
# stride = (1,1)
# poolingShape = (3,3)
# input_tensor = np.asarray([[12,3,4,5],[4,12,27,13],[5,6,9,85],[12,54,65,8]])
# # print(n)
# # # print(skimage.measure.block_reduce(n, (2,2), np.max))
#
# # stride = (2,2)
# # poolingShape=(2,2)
#
# output_shape = (input_tensor.shape[0] - poolingShape[0] + 1,
#                 input_tensor.shape[1] - poolingShape[1] + 1)
#
# out = np.zeros(output_shape)
# out_index = []
#
# for i in range(0,input_tensor.shape[0]-poolingShape[0]+1):
#     print(i)
#     for j in range(0,input_tensor.shape[1]-poolingShape[1]+1):
#         max = 0
#
#         for x in np.arange(i,i+poolingShape[0]):
#             for y in np.arange(j,j+poolingShape[1]):
#                 if input_tensor[x,y] > max:
#                     max = input_tensor[x,y]
#                     out = []
#                     index = (x,y)
#         indxxx.append(index)
#
#
#
#
#
# print(indxxx,maxxx,len(maxxx))
#



for i in range(0,n.shape[0]-poolingShape[0]+1):
    for j in range(0,n.shape[1]-poolingShape[1]+1):
        max = 0
        #index = np.zeros((2,1))

        for x in np.arange(i,i+poolingShape[0]):
            for y in np.arange(j,j+poolingShape[1]):
                if n[x,y] > max:
                    max = n[x,y]
                    # print(max)
                    index = (x,y)
        indxxx.append(index)
        maxxx.append(max)
shape = ((n.shape[0]-poolingShape[0])//1 + 1, (n.shape[1]-poolingShape[1])//1 + 1)
maxxx = np.asarray(maxxx)
out = maxxx.reshape(shape)

#####################################################################################################################


print(out[::stride[1],::stride[0]],len(maxxx))