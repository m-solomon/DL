import torch as t
from src_to_implement.trainer import Trainer
import sys
import torchvision as tv


from src_to_implement.model.resnet import ResNet #I added this


epoch = int(sys.argv[1])
#TODO: Enter your model here

model = ResNet()

crit = t.nn.BCEWithLogitsLoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
