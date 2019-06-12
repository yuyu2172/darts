# import torch
# import torch.nn as nn
import chainer
import chainer.functions as F
import chainer.links as L


OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: Pooling2D(
      'average', 3, stride=stride, pad=1),
  'max_pool_3x3' : lambda C, stride, affine: Pooling2D(
      'max', 3, stride=stride, pad=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: ConvKx1And1xK(C, C, 7, stride, 3, affine)
}


class Pooling2D(object):

    def __init__(self, op, ksize, stride, pad):
        assert op in ['average', 'max']
        self.op = op
        self.ksize = ksize
        self.stride = stride
        self.pad = pad

    def __call__(self, x):
        if self.op == 'average':
            # count_include_pad = False
            return F.average_pooling_2d(
                    x, self.ksize, self.stride, self.pad)
        elif self.op == 'max':
            return F.max_pooling_2d(
                    x, self.ksize, self.stride, self.pad)


class ConvKx1And1xK(chainer.Sequential):

    def __init__(
            self, in_C, out_C, ksize, stride, pad, affine=True):
        assert in_C == out_C
        bn_kwargs = {'eps': 1e-05}
        if not affine:
            bn_kwargs.update({'use_gamma': False, 'use_gamma': False})
        links = [F.relu,
                 L.Convolution2D(in_C, in_C, (1, ksize), (1, stride), (0, pad),
                                 nobias=True),
                 L.Convolution2D(in_C, out_C, (ksize, 1), (stride, 1), (pad, 0),
                                 nobias=True),
                 L.BatchNormalization(out_C, **bn_kwargs)
                 ]
        super(ConvKx1And1xK, self).__init__(*links)


class ReLUConv2DBN(chainer.Sequential):

    def __init__(
            self, in_C, out_C, ksize, stride, pad, affine=True):
        bn_kwargs = {'eps': 1e-05}
        if not affine:
            bn_kwargs.update({'use_gamma': False, 'use_gamma': False})
        links = [F.relu,
                 L.Convolution2D(in_C, out_C, ksize, stride, pad,
                                 nobias=True),
                 L.BatchNormalization(out_C, **bn_kwargs)
                 ]
        super(ReLUConv2DBN, self).__init__(*links)


class DilConv(chainer.Sequential):

    def __init__(
            self, in_C, out_C, ksize, stride, pad, dilate, affine=True):
        bn_kwargs = {'eps': 1e-05}
        if not affine:
            bn_kwargs.update({'use_gamma': False, 'use_gamma': False})
        links = [F.relu,
                 L.Convolution2D(
                     in_C, in_C, ksize, stride, pad,
                     dilate=dilate, groups=in_C, nobias=True),
                 L.Convolution2D(
                     in_C, out_C,
                     ksize=1, stride=1, pad=0, nobias=True),
                 L.BatchNormalization(out_C, **bn_kwargs)
                 ]
        super(DilConv, self).__init__(*links)


class SepConv(chainer.Sequential):

    def __init__(
            self, in_C, out_C, ksize, stride, pad, affine=True):
        bn_kwargs = {'eps': 1e-05}
        if not affine:
            bn_kwargs.update({'use_gamma': False, 'use_gamma': False})

        links = [F.relu,
                 L.Convolution2D(
                     in_C, in_C, ksize, stride, pad,
                     groups=in_C, nobias=True),
                 L.Convolution2D(
                     in_C, out_C, ksize=1, stride=1, pad=0, nobias=True),
                 L.BatchNormalization(in_C, **bn_kwargs),
                 F.relu,
                 L.Convolution2D(
                     in_C, in_C, ksize, stride=1, pad=pad,
                     groups=in_C, nobias=True),
                 L.Convolution2D(
                     in_C, out_C, ksize=1, stride=1, pad=0, nobias=True),
                 L.BatchNormalization(out_C, **bn_kwargs)
                 ]
        super(SepConv, self).__init__(*links)


class Identity(chainer.Sequential):

    def __init__(self):
        links = [lambda x: x]
        super(Identity, self).__init__(*links)


class Zero(chainer.Chain):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0
        else:
            return x[:, :, ::self.stride, ::self.stride] * 0


class FactorizedReduce(chainer.Chain):

    def __init__(self, in_C, out_C, affine=True):
        bn_kwargs = {'eps': 1e-05}
        if not affine:
            bn_kwargs.update({'use_gamma': False, 'use_gamma': False})

        super(FactorizedReduce, self).__init__()
        assert out_C % 2 == 0
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                    in_C, out_C // 2, ksize=1, stride=2, pad=0, nobias=True)
            self.conv2 = L.Convolution2D(
                    in_C, out_C // 2, ksize=1, stride=2, pad=0, nobias=True)
            self.bn = L.BatchNormalization(out_C, **bn_kwargs)

    def forward(self, x):
        h = F.relu(x)
        h = F.concat([self.conv1(h), self.conv2(h[:, :, 1:, 1:])], axis=1)
        h = self.bn(h)
        return h
