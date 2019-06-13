import argparse
import math
import chainer
from chainer.training import extensions

from darts.links.model import NetworkCIFAR
from darts.genotypes import DARTS

from darts.cifar_transforms import cifar10_val_transform
from darts.cifar_transforms import CIFAR10TrainTransform
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv.chainer_experimental.training.extensions import make_shift
from darts.links.model import TrainChain


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batchsize', type=int, default=96)
    parser.add_argument('--epoch', type=int, default=600)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--use-cutout', action='store_true')
    parser.add_argument('--cutout-length', type=int, default=16)
    parser.add_argument('--init-channels', type=int, default=36)
    parser.add_argument('--n-layer', type=int, default=20)
    parser.add_argument('--use-auxiliary', action='store_true')
    parser.add_argument('--auxiliary-weight', type=float, default=0.4)
    parser.add_argument('--drop-path-prob', type=float, default=0.2)
    args = parser.parse_args()

    model = NetworkCIFAR(
        DARTS, init_C=args.init_channels,
        n_class=10, n_layer=args.n_layer,
        use_auxiliary=args.use_auxiliary,
        drop_path_prob=args.drop_path_prob)
    classifier = TrainChain(model, args.use_auxiliary)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        classifier.to_gpu()
    
    optimizer = chainer.optimizers.CorrectedMomentumSGD(args.lr, args.momentum)
    optimizer.setup(classifier)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(3e-4))

    train_data, val_data = chainer.datasets.get_cifar10()
    train_data = TransformDataset(
        train_data, ('img', 'label'),
        CIFAR10TrainTransform(args.use_cutout, args.cutout_length))
    val_data = TransformDataset(
        val_data, ('img', 'label'), cifar10_val_transform)

    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, args.batchsize, n_prefetch=1)
    val_iter = chainer.iterators.MultiprocessIterator(val_data, args.batchsize,
                                                      repeat=False, shuffle=False)
    evaluator = extensions.Evaluator(val_iter, classifier, device=args.gpu)

    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(evaluator, trigger=(1, 'epoch'))

    @make_shift('lr')
    def lr_schedule(trainer):
        max_lr = args.lr
        min_lr = 0

        epoch = trainer.updater.epoch_detail
        progress_ratio = epoch / args.epoch
        rate = 0.5 * (math.cos(math.pi * progress_ratio) + 1)
        return min_lr + max_lr * rate

    trainer.extend(lr_schedule)
    trainer.extend(extensions.LogReport(), trigger=(1, 'epoch'))

    trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'lr', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']),
        trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=50))
    trainer.extend(
        extensions.snapshot_object(
            model, 'model_iter_{.updater.epoch_detail}'),
        trigger=(50, 'epoch'))

    trainer.run()
