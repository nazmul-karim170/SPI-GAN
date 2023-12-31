# -*- coding: utf-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802

TODO:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from vgg_mod import vgg16 as vgg16_mod
from vgg import VGG
import numpy as np
from functools import reduce

def swish(x):
    return x * F.sigmoid(x)

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        feature = cnn.state_dict()      ## To get the neural network weights
        features_1 = feature['features.0.weight']

        ### Graysacale Feature Conversion
        features_vgg = torch.from_numpy(np.average(np.array(features_1),1, weights =[0.299, 0.587, 0.144])).view(64,1,3,3)     ## Get the feature for grayscale image (weighted average)

        feature['features.0.weight'] = features_vgg
        vgg_mod1 = VGG('VGG19')
        vgg_mod1.load_state_dict(feature)
        self.features = nn.Sequential(*list(vgg_mod1.features)[:11]).eval()

    def forward(self, x):
        return self.features(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()
        self.relu =  nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels, 64, 5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.upsample_block  = upsampleBlock(64,64)
        # self.conv3 = nn.Conv2d(32, 1, 5, stride=1, padding=2)
        # self.bn3 = nn.BatchNorm2d(1)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        # return self.bn2(self.conv2(y)) + x
        return  self.upsample_block(y) + x

class upsampleBlock(nn.Module):

    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(1)
        self.relu =  nn.PReLU()

    def forward(self, x):
        return self.relu(self.shuffler(self.conv(x)))    

class PreGenerator(nn.Module):
    
    def __init__(self, n_residual_blocks, upsample_factor):
        super(PreGenerator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.relu =  nn.PReLU()
        self.conv1 = nn.Conv2d(1, 128, 9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, 5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        return x

class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor
        self.relu =  nn.PReLU()

        self.conv1_p = nn.Conv2d(1, 128, 9, stride=1, padding=4)
        self.bn1_p   = nn.BatchNorm2d(128)
        self.conv2_p = nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.bn2_p   = nn.BatchNorm2d(64)
        self.conv3_p = nn.Conv2d(64, 1, 5, stride=1, padding=2)
        self.bn3_p   = nn.BatchNorm2d(1)

        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(64)  

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        # for i in range(int(self.upsample_factor/2)):
        #     self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.conv3_2 = nn.Conv2d(64, 1, 9, stride=1, padding=4)

    def forward(self, x1):
        
        x = self.relu(self.bn1_p(self.conv1_p(x1)))
        x = self.relu(self.bn2_p(self.conv2_p(x)))
        x = self.bn3_p(self.conv3_p(x))
        
        x = self.relu(self.bn1(self.conv1(x)))
        # x = self.upsample_block(x)
        # x = nn.PReLU(self.bn2(self.conv2(x)))
        # x = self.bn3(self.conv3(x))

        y = x.clone()           ## .clone() ??
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.bn2_2(self.conv2_2(y)) + x1

        # for i in range(int(self.upsample_factor/2)):
        #     x = self.__getattr__('upsample' + str(i+1))(x)

        return self.conv3_2(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        # self.conv9 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        # self.bn9 = nn.BatchNorm2d(1024)
        # self.conv10= nn.Conv2d(1024, 1024, 3, stride=2, padding=1)
        # self.bn10 = nn.BatchNorm2d(1024)

        ## Replaced original paper FC layers with FCN
        self.conv11 = nn.Conv2d(512, 1, 1, stride=1, padding=1)
        self.relu =  nn.LeakyReLU(0.02)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        # x = self.relu(self.bn9(self.conv9(x)))
        # x = self.relu(self.bn10(self.conv10(x))) 

        x = self.conv11(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)
