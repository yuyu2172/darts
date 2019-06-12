import argparse
import torch
import chainer


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


weight = torch.load('cifar10_model.pt')
