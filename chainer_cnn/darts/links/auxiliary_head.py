import chainer.functions as F
import chainer.links as L
from chainercv.links import PickableSequentialChain
from ..operations import Pooling2D


class AuxiliaryHeadCIFAR(PickableSequentialChain):

    def __init__(self, C, n_class):
        bn_kwargs = {'eps': 1e-05}
        super(AuxiliaryHeadCIFAR, self).__init__()
        with self.init_scope():
            self.relu1 = F.relu 
            self.pool1 = Pooling2D('average', 5, 3, 0)  # img size = 2x2
            self.conv2 = L.Convolution2D(C, 128, 1, nobias=True)
            self.bn2 = L.BatchNormalization(128, **bn_kwargs)
            self.relu2 = F.relu
            self.conv3 = L.Convolution2D(128, 768, 2, nobias=True)
            self.bn3 = L.BatchNormalization(768, **bn_kwargs)
            self.relu3 = F.relu
            self.classifier = L.Linear(768, n_class)
