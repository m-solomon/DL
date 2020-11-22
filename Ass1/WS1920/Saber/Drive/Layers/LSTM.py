import numpy as np
from Layers import Base
from Layers import FullyConnected
from Layers.TanH import *
from Layers.Sigmoid import *
from Layers.FullyConnected import *
from Layers.Initializers import *
import copy



class LSTM(Base.Base):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(phase="train")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.Fully_input_size = self.hidden_size + self.input_size

        self.memorize = False
        self.optimizer= None

        # self.weights = None

        self.hidden_parameters = np.zeros(self.hidden_size)
        self.fc_HIDDEN_obj = FullyConnected(self.Fully_input_size, 4 * self.hidden_size)
        self.fc_OUT_obj = FullyConnected(self.hidden_size, self.output_size)




# **********************************************************************************************************************#

    def forward(self, input_tensor):


        self.FC_concatenated_input = np.zeros((input_tensor.shape[0], self.input_size + self.hidden_size))
        self.input_tensor = np.asarray(input_tensor)
        self.batch_size = input_tensor.shape[0]
        self.C_parameters = np.zeros((self.hidden_size,1))

        self.output_layer_activation = np.zeros((self.batch_size, self.output_size))
        self.TanH_out_cache = np.zeros((self.batch_size, self.hidden_size))

        if self.memorize == False:
            self.hidden_parameters = np.zeros(self.hidden_size) #do we update it in the loop?


        self.Y = np.zeros((self.batch_size, self.output_size))

        self.cache = np.zeros((6, self.batch_size + 1, self.hidden_size, 1))
        self.cache[5, 0] = self.hidden_parameters.reshape(self.hidden_size,1)

        for i in range(self.batch_size):

            input_Fc_input = self.input_tensor[int(i)].reshape(input_tensor.shape[1],1)
            input_Fc_param = self.hidden_parameters.reshape(self.hidden_size,1)

            self.Fully_input = np.concatenate((input_Fc_param, input_Fc_input))
            self.FC_concatenated_input[i][:, np.newaxis] = self.Fully_input


            Fully_connected_HIDDEN = self.fc_HIDDEN_obj.forward(self.Fully_input.T)




            #instantiation of the activation function
            #hidden layer
            self.sigmoid_ft = Sigmoid()
            self.sigmoid_it = Sigmoid()
            self.TanH_Ctelt = TanH()
            self.sigmoid_ot = Sigmoid()
            #outer layer
            self.sigmoid_outer = Sigmoid()
            #red TanH
            self.red_TanH = TanH()

            self.ft = self.sigmoid_ft.forward(Fully_connected_HIDDEN[0][0:self.hidden_size]).reshape(self.hidden_size,1)
            self.it = self.sigmoid_it.forward(Fully_connected_HIDDEN[0][self.hidden_size:2* self.hidden_size]).reshape(self.hidden_size,1)
            self.Ctelt = self.TanH_Ctelt .forward(Fully_connected_HIDDEN[0][2*self.hidden_size:3*self.hidden_size]).reshape(self.hidden_size,1)
            self.ot = self.sigmoid_ot.forward(Fully_connected_HIDDEN[0][3*self.hidden_size:4*self.hidden_size]).reshape(self.hidden_size,1)

            self.C_parameters = self.ft * self.C_parameters + self.Ctelt * self.it
            temp1 = self.red_TanH.forward(self.C_parameters)
            self.hidden_parameters = temp1 * self.ot

            self.TanH_out_cache[i][:] = temp1.reshape(self.hidden_size)

            self.cache[0, i+1] = self.ft
            self.cache[1, i+1] = self.it
            self.cache[2, i+1] = self.Ctelt
            self.cache[3, i+1] = self.ot
            self.cache[4, i+1] = self.C_parameters
            self.cache[5, i+1] = self.hidden_parameters


            self.output_layer_activation[i,:] = self.fc_OUT_obj.forward(self.hidden_parameters.T)
            self.Y[i] = self.sigmoid_outer.forward(self.fc_OUT_obj.forward(self.hidden_parameters.T))

        return self.Y


    def backward(self, error_tensor):



        passed_error_tensor = np.zeros((error_tensor.shape[0], self.input_size))

        self.error_tensor = error_tensor
        self.C_parameters_grad = np.zeros((self.hidden_size,1))
        self.hidden_parameters_grad = np.zeros((self.hidden_size,1))
        batch_size = self.error_tensor.shape[0]

        grad_OUT = 0.0 #np.zeros((batch_size,self.error_tensor.shape[1]))
        grad_HIDD = 0.0  #np.zeros((batch_size, self.input_size + self.hidden_size))

        for t in reversed(range(batch_size)):
            ft = self.cache[0, t + 1]
            it = self.cache[1, t + 1]
            Ctelt = self.cache[2, t + 1]
            ot = self.cache[3, t + 1]
            c_pre = self.cache[4, t]
            c_nxt = self.cache[4, t+1]
            a_pre = self.cache[5, t]
            a_nxt = self.cache[5, t+1]


            # the gradient of the sigmoid function at the output
            outer_sigmoid_gradient = self.sigmoid_outer.backward(error_tensor[t])

            # the gradient of the weights of the outer layer
            temp = self.fc_OUT_obj.backward(outer_sigmoid_gradient)
            grad_OUT += (self.fc_OUT_obj.dWeights)

            # the gradient at star 1   = gradient of the incoming hidden gradient + gradient of the weights of the outer layer
            node_gradient_1 = temp.T + self.hidden_parameters_grad

            # gradient of ot = the node gradient * output of the red TanH
            tempp = (self.TanH_out_cache[t:t + 1, :])
            d_ot = node_gradient_1 * tempp.T

            # the gradient at the output of the red TanH.... = ot * the node gradient
            TanH_node = node_gradient_1 * ot

            # the gradient of the red TanH
            TanH_gradient = self.red_TanH.backward(TanH_node)



            # the gradient at node star 2  = the gradient of TanH + the gradient of the incoming C_parameters
            node_gradient_2 = self.C_parameters_grad + TanH_gradient

            # the gradient of C_teld
            d_Ctelt = node_gradient_2 * it

            # the gradient of it
            d_it = node_gradient_2 * Ctelt


            # the gradient of ft
            d_ft = node_gradient_2 * c_pre #### CHECK THIS AGAIN !!!!!!!!

            # the gradient of the backprobagating c parameter
            self.C_parameters_grad = node_gradient_2 * ft


            # the gradients of the activations of the hidden layers
            sigmoid_ot_gradient = self.sigmoid_ot.backward(d_ot)
            TanH_Ctelt_gradient = self.TanH_Ctelt.backward(d_Ctelt)
            sigmoid_it_gradient = self.sigmoid_ot.backward(d_it)
            sigmoid_ft_gradient = self.sigmoid_ot.backward(d_ft)

            # Next step, concatenate and call Hidden fully connected backward method
            concatenated_activation_error = np.concatenate((sigmoid_ft_gradient, sigmoid_it_gradient, TanH_Ctelt_gradient, sigmoid_ot_gradient), axis = 0)

            # we call backward on the concatenated tensor to obtain the WEIGHTS GRADIENT and the passed dh and the returned errortensor
            temp2 = self.fc_HIDDEN_obj.backward(concatenated_activation_error.T)
            grad_HIDD += (self.fc_HIDDEN_obj.dWeights)
            self.hidden_parameters_grad = temp2.T[0:self.hidden_size]
            passed_error_tensor[t][:,np.newaxis] = temp2.T[self.hidden_size::]


        # transform the weight gradient lists into arrays
        self.dweights_outer = np.asarray(grad_OUT)
        self.dweights_hidden = np.asarray(grad_HIDD)
        m = self.fc_HIDDEN_obj.weights
        if (self.optimizer != None):
            self.fc_HIDDEN_obj.weights = self.optimizer.calculate_update(self.fc_HIDDEN_obj.weights, self.dweights_hidden)
            self.fc_OUT_obj.weights = self.optimizer1.calculate_update(self.fc_OUT_obj.weights, self.dweights_outer)

        n = self.fc_HIDDEN_obj.weights
        return passed_error_tensor



    @property
    def memorizes(self):
        return self.memorize

    @memorizes.setter
    def memorizes(self, var):
        self.memorize = var

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = copy.deepcopy(optimizer)

    @property
    def optimizer1(self):
        return self.__optimizer

    @optimizer1.setter
    def optimizer1(self, optimizer1):
        self.__optimizer = copy.deepcopy(optimizer1)

    def initialize(self, weights_initializer, bias_initializer):

        self.fc_HIDDEN_obj.initialize(weights_initializer, bias_initializer)
        self.fc_OUT_obj.initialize(weights_initializer, bias_initializer)

    @property
    def weights(self):
        return self.fc_HIDDEN_obj.return_w()

    @weights.setter
    def weights(self, w):
        self.fc_HIDDEN_obj.weights = w



    @property
    def gradient_weights(self):
        return self.fc_HIDDEN_obj.gradient_weights
