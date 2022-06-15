# This file is modified from official pycls repository

"""Model and loss construction functions."""

from pycls.core.net import SoftCrossEntropyLoss
from pycls.models.resnet import *
from pycls.models.vgg import *
from pycls.models.alexnet import *

import torch
from torch import nn
from torch.nn import functional as F
# Supported models
_models = {
    # VGG style architectures
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19': vgg19,
    'vgg19_bn': vgg19_bn,

    # ResNet style archiectures
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,

    # AlexNet architecture
    'alexnet': alexnet
}

# Supported loss functions
_loss_funs = {"cross_entropy": SoftCrossEntropyLoss}


class FeaturesNet(nn.Module):
    def __init__(self, in_layers, out_layers, use_mlp=False, penultimate_active=False):
        super().__init__()
        self.use_mlp = use_mlp
        self.penultimate_active = penultimate_active
        self.lin1 = nn.Linear(in_layers, in_layers)
        self.lin2 = nn.Linear(in_layers, in_layers)
        self.final = nn.Linear(in_layers, out_layers)

    def forward(self, x):
        feats = x
        if self.use_mlp:
            x = F.relu(self.lin1(x))
            x = F.relu((self.lin2(x)))
        out = self.final(x)
        if self.penultimate_active:
            return feats, out
        return out


def get_model(cfg):
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]


def get_loss_fun(cfg):
    """Gets the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]


def build_model(cfg):
    """Builds the model."""
    if cfg.MODEL.LINEAR_FROM_FEATURES:
        num_features = 384 if cfg.DATASET.NAME in ['IMAGENET50', 'IMAGENET100', 'IMAGENET200'] else 512
        return FeaturesNet(num_features, cfg.MODEL.NUM_CLASSES)

    model = get_model(cfg)(num_classes=cfg.MODEL.NUM_CLASSES, use_dropout=True)
    if cfg.DATASET.NAME == 'MNIST':
        model.conv1 =  torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    return model 


def build_loss_fun(cfg):
    """Build the loss function."""
    return get_loss_fun(cfg)()


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
