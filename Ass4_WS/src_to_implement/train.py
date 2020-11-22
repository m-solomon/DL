import torch as t
import torchvision as tv
from src_to_implement.data import get_train_dataset, get_validation_dataset
from src_to_implement.stopping import EarlyStoppingCallback
from src_to_implement.trainer import Trainer
from src_to_implement.model.resnet import ResNet #added this
from src_to_implement.model.resnet import BasicBlock #added this
from matplotlib import pyplot as plt
import numpy as np




trainloader = t.utils.data.DataLoader(get_train_dataset())
testloader = t.utils.data.DataLoader(get_validation_dataset())

nett = ResNet(BasicBlock)

crit = t.nn.BCEWithLogitsLoss(pos_weight=get_validation_dataset().pos_weight())

optimizer = t.optim.SGD(nett.parameters(), lr=0.001, momentum=0.9)
early_stop = EarlyStoppingCallback(2)

train = Trainer(nett, crit, optimizer, trainloader, testloader, cuda=True, early_stopping_cb = early_stop)





res = train.fit(2)

t.restore_checkpoint(2)
t.save_onnx('checkpoint_{:03d}.onnx'.format(2))


#TODO

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')