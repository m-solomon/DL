import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import *
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO

# create an instance of our ResNet model
# TODO

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO

# go, go, go... call fit on trainer
#res = #TODO
###################################################################################################
###################################################################################################
DF = pd.read_csv("C:\\Users\\utg_1\\OneDrive\\Documents\\Mohamed's\\Studies\\FAU\\SS20\\DL\\Ass4\\src_to_implement\\data.csv")
train_dataset, val_dataset = train_test_split(DF, test_size=0.33 )

Train = ChallengeDataset(train_dataset , "train")

Val  = ChallengeDataset(val_dataset , "val")

train_load = t.utils.data.DataLoader(Train, batch_size=32 )
val_load = t.utils.data.DataLoader(Val, batch_size=32)


net = ResNet(BasicBlock)

crit = t.nn.BCEWithLogitsLoss()

optimizer = t.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train = Trainer(net, crit, optimizer, train_load, val_load, cuda=False)




res = train.fit(12)

train.restore_checkpoint(12)
train.save_onnx("C:\\Users\\utg_1\\OneDrive\\Documents\\Mohamed's\\Studies\\FAU\\SS20\\DL\\Ass4\\src_to_implement\\checkpoint_{:03d}.onnx".format(2))


###################################################################################################
###################################################################################################


# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')