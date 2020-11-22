import torch as t
from trainer import Trainer
import sys
import torchvision as tv
from model import *

epoch = 10
#TODO: Enter your model here
model = ResNet(BasicBlock)

crit = t.nn.BCEWithLogitsLoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_out_{:03d}.onnx'.format(epoch))
