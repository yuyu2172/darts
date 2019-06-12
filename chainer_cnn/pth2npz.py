import argparse
import numpy as np
import torch
import chainer

from darts.links.model import NetworkCIFAR
from darts.genotypes import DARTS

from darts.operations import *


def copy_linear(l, key, val, bias=False):
    print(key)
    if bias:
        l.b.data[:] = val.cpu().numpy()
    else:
        l.W.data[:] = val.cpu().numpy()

def copy_conv(l, key, val, bias=False, flip=False):
    print(key)
    if bias:
        l.b.data[:] = val.cpu().numpy()
    else:
        if flip:
            l.W.data[:] = val.cpu().numpy()[:, ::-1]
        else:
            l.W.data[:] = val.cpu().numpy()


def copy_bn(l, key, val):
    print(key)
    if key[-6:] == 'weight':
        l.gamma.data[:] = val.cpu().numpy()
    elif key[-4:] == 'bias':
        l.beta.data[:] = val.cpu().numpy()
    elif key[-12:] == 'running_mean':
        l.avg_mean.data[:] = val.cpu().numpy()
    elif key[-11:] == 'running_var':
        l.avg_var.data[:] = val.cpu().numpy()

dummy = np.zeros((1, 3, 32, 32), dtype=np.float32)

model = NetworkCIFAR(DARTS)
model(dummy)
weight = torch.load('cifar10_model.pt')
keys = list(weight.keys())

for key, val in weight.items():
    s_keys = key.split('.')
    if s_keys[0] == 'stem':
        if s_keys[1] == '0':
            copy_conv(model.stem.conv, key, val)
        elif s_keys[1] == '1':
            copy_bn(model.stem.bn, key, val)
    elif s_keys[0] == 'cells':
        l = model.cells[int(s_keys[1])]
        pre = False
        if s_keys[2] == 'preprocess0':
            l = getattr(l, 'preprocess0')
            pre = True
        elif s_keys[2] == 'preprocess1':
            l = getattr(l, 'preprocess1')
            pre = True
        if pre:
            if s_keys[4] == '1':
                copy_conv(l[1], key, val)
            elif s_keys[4] == '2':
                copy_bn(l[2], key, val)
            elif s_keys[3] == 'conv_1':
                copy_conv(l.conv1, key, val)
            elif s_keys[3] == 'conv_2':
                copy_conv(l.conv2, key, val)
            elif s_keys[3] == 'bn':
                copy_bn(l.bn, key, val)
            else:
                raise ValueError(key)
        else:
            if s_keys[2] == '_ops':
                l = l.ops[int(s_keys[3])]
                if s_keys[5] == '1':
                    copy_conv(l[1], key, val)
                elif s_keys[5] == '2':
                    copy_conv(l[2], key, val)
                elif s_keys[5] == '3':
                    copy_bn(l[3], key, val)
                elif s_keys[5] == '5':
                    copy_conv(l[5], key, val)
                elif s_keys[5] == '6':
                    copy_conv(l[6], key, val)
                elif s_keys[5] == '7':
                    copy_bn(l[7], key, val)
    elif s_keys[0] == 'auxiliary_head':
        if s_keys[1] == 'features':
            if s_keys[2] == '2':
                copy_conv(model.auxiliary_head[2], key, val)
            elif s_keys[2] == '3':
                copy_bn(model.auxiliary_head[3], key, val)
            elif s_keys[2] == '5':
                copy_conv(model.auxiliary_head[5], key, val)
            elif s_keys[2] == '6':
                copy_bn(model.auxiliary_head[6], key, val)
        elif s_keys[1] == 'classifier':
            if s_keys[-1] == 'weight':
                copy_linear(model.auxiliary_head[8], key, val)
            elif s_keys[-1] == 'bias':
                copy_linear(model.auxiliary_head[8], key, val, bias=True)

    elif s_keys[0] == 'classifier':
        if s_keys[-1] == 'weight':
            copy_linear(model.classifier, key, val)
        elif s_keys[-1] == 'bias':
            copy_linear(model.classifier, key, val, bias=True)
    else:
        raise ValueError(key)

chainer.serializers.save_npz('model.npz', model)

