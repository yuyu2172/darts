import argparse
import numpy as np
import torch
import chainer

from darts.links.model import NetworkCIFAR
from darts.genotypes import DARTS

from darts.operations import *

import sys
sys.path.append('../cnn')
from model import NetworkCIFAR as Network
from chainer.datasets import get_cifar10


c_model = NetworkCIFAR(DARTS)
chainer.serializers.load_npz('model.npz', c_model)
chainer.global_config.train = False


t_model = Network(36, 10, 20, True, DARTS)
t_model.drop_path_prob = 0.2
t_model.load_state_dict(torch.load('cifar10_model.pt'))
t_model.eval()


train, val = get_cifar10()
img, _ = train[0]
x = img[None]
t_x = torch.autograd.Variable(torch.FloatTensor(x))
c_y = c_model(x)
t_y = t_model(t_x)
np.testing.assert_almost_equal(
    c_y[0].data, t_y[0].detach().data.numpy(), decimal=5)



#####################
# c_prev_s = c_model.stem(x)
# c_prev_prev_s = c_prev_s
# for i in range(7):
#     c_prev_prev_s, c_prev_s = c_prev_s, c_model.cells[i](
#         c_prev_prev_s, c_prev_s, 0.1)
# 
# t_prev_s = t_model.stem(t_x)
# t_prev_prev_s = t_prev_s
# for i in range(7):
#     t_prev_prev_s, t_prev_s = t_prev_s, t_model.cells[i](
#         t_prev_prev_s, t_prev_s, 0.1)
# 
# np.testing.assert_almost_equal(c_prev_s.data, t_prev_s.detach().data.numpy(), decimal=5)
