import chainer.functions as F
import chainer.links as L
import chainer
from ..operations import Pooling2D


class AuxiliaryHeadCIFAR(chainer.Sequential):

    def __init__(self, C, n_class):
        bn_kwargs = {'eps': 1e-05}

        links = [F.relu,
                 Pooling2D('average', 5, 3, 0),  # img size = 2x2
                 L.Convolution2D(C, 128, 1, nobias=True),
                 L.BatchNormalization(128, **bn_kwargs),
                 F.relu,
                 L.Convolution2D(128, 768, 2, nobias=True),
                 L.BatchNormalization(768, **bn_kwargs),
                 F.relu,
                 L.Linear(768, n_class)
                 ]
        super(AuxiliaryHeadCIFAR, self).__init__(*links)
