import torch as t
import torchvision as tv
# from src_to_implement.data import get_train_dataset, get_validation_dataset
# #
# # from src_to_implement.stopping import EarlyStoppingCallback
# # from src_to_implement.trainer import Trainer
# # from src_to_implement.model.resnet import ResNet #added this
# # from src_to_implement.model.resnet import BasicBlock #added this

from data import get_train_dataset, get_validation_dataset
from stopping import EarlyStoppingCallback
from trainer import Trainer
from model.resnet import ResNet #added this
from model.resnet import resnet34 #added this
from model.resnet import BasicBlock #added this
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.models as models


trainloader = t.utils.data.DataLoader(get_train_dataset(), batch_size = 64)
testloader = t.utils.data.DataLoader(get_validation_dataset(), batch_size = 64)

def set_parameter_requires_grad(model, feature_extracting): ## added this
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# nett = ResNet(BasicBlock)
nett = models.resnet34(pretrained=True)
##
set_parameter_requires_grad(nett, True)
num_ftrs = nett.fc.in_features
nett.fc = nn.Linear(num_ftrs, 2)

pw = get_validation_dataset().pos_weight()
crit = t.nn.BCEWithLogitsLoss(pos_weight=get_validation_dataset().pos_weight())
optimizer = t.optim.Adam(nett.parameters(), lr=0.001, weight_decay=0.00001)
early_stop = EarlyStoppingCallback(5)
train = Trainer(nett, crit, optimizer, trainloader, testloader, cuda=True, early_stopping_cb = early_stop)

# go, go, go... call fit on trainer

res = train.fit(60)
# train.restore_checkpoint(2)
# train.save_onnx('checkpoint_{:03d}.onnx'.format(2))
#TODO

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')