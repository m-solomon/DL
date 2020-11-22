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
####Parameters:
ep = 20
BZ = 100
TZ = 0.4
LR = 0.0005
opt = 'A'
DF = pd.read_csv('data.csv')
train_dataset, val_dataset = train_test_split(DF, test_size= TZ)

Train = ChallengeDataset(train_dataset , "train")

Val  = ChallengeDataset(val_dataset , "val")

train_load = t.utils.data.DataLoader(Train, batch_size=BZ )
val_load = t.utils.data.DataLoader(Val, batch_size=BZ )


net = ResNet(BasicBlock)

crit = t.nn.BCELoss()                #no sigmoid inside
#crit = t.nn.BCEWithLogitsLoss()       #with sigmoid inside -->must comment sigmoid in model

if opt == 'A':
    optimizer = t.optim.Adam(net.parameters(), lr=LR)
else:
    optimizer = t.optim.SGD(net.parameters(), lr=LR, momentum=0.9)


train = Trainer(net, crit, optimizer, train_load, val_load, cuda=True)




res = train.fit(ep)

train.restore_checkpoint(ep)
train.save_onnx('checkpoint_{:03d}.onnx'.format(ep))


###################################################################################################
###################################################################################################


# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')