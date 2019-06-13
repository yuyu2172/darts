import argparse
import numpy as np
import chainer

from darts.links.model import NetworkCIFAR
from darts.genotypes import DARTS

from darts.operations import *

from darts.cifar_transforms import cifar10_val_transform
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from darts.links.model import TrainChain


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-model', type=str, default='model.npz')
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    model = NetworkCIFAR(DARTS)
    chainer.serializers.load_npz(args.pretrained_model, model)
    chainer.global_config.train = False
    classifier = TrainChain(model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        classifier.to_gpu()

    _, val = chainer.datasets.get_cifar10()
    val = TransformDataset(
        val, ('img', 'label'), cifar10_val_transform)

    it = chainer.iterators.SerialIterator(val, 32, False, False)
    evaluator = chainer.training.extensions.Evaluator(
        it, classifier, device=args.gpu)
    result = evaluator()
    print('Top 1 error {}%'.format(100 * float(1 - result['main/accuracy'])))
