import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class BasicBlock(nn.Module):


    def __init__(self, input_ch, outout_ch, stride): #stride = 1
        super(BasicBlock, self).__init__()
        self.stride = stride


        self.conv1 = nn.Conv2d(input_ch, outout_ch, kernel_size=3, stride=self.stride, padding =1 ) #padding =0 
        self.bn1 = nn.BatchNorm2d(outout_ch)

        self.relu= nn.ReLU()       #inplace=True

        self.conv2 = nn.Conv2d(outout_ch, outout_ch, kernel_size=3, padding =1)  #no stride is used #padding =1
        self.bn2 = nn.BatchNorm2d(outout_ch)
        
        self.conv_id = nn.Conv2d(input_ch, outout_ch, kernel_size=1, stride=self.stride)  #no stride is used #padding =1
        self.bn_id = nn.BatchNorm2d(outout_ch)

        
    def forward(self, x):
        identity = x.clone()
        identity = self.conv_id(identity)
        identity = self.bn_id(identity)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += identity

        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_classes=2):
        super(ResNet, self).__init__()
        #self.in_chanels = 64
        self.image_chanels = 3

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2) #padding = 3 bias= False
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU() #inplace=True
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2) #padding = 1
        self.layer1 = block(64, 64, 1) #3
        self.layer2 = block(64, 128, 2) #4
        self.layer3 = block(128, 256, 2) #6
        self.layer4 = block(256, 512, 2) #3
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # (7, stride = 1)       # nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes) # instead of 64 -> 16*n_features
        #self.sig = nn.Sigmoid()

    def forward(self, x): # CLEAR
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = self.sig(x)
        

        return x