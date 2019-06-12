import numpy as np

from chainercv.transforms import random_crop
from chainercv.transforms import random_flip


CIFAR_MEAN = np.array([0.49139968, 0.48215827, 0.44653124], dtype=np.float32)
CIFAR_STD = np.array([0.24703233, 0.24348505, 0.26158768], dtype=np.float32)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        # mask = torch.from_numpy(mask)
        # mask = mask.expand_as(img)
        img *= mask[None]
        return img


class CIFAR10TrainTransform(object):

    def __init__(self, do_cutout, cutout_length):
        self.do_cutout = do_cutout
        self.cutout_length = cutout_length

    def __call__(self, in_data):
        img, label = in_data

        img = np.pad(img, ((0, 0), (4, 4), (4, 4)), 'constant')
        img = random_crop(img, (32, 32), x_random=True)
        img = (img - CIFAR_MEAN[:, None, None]) / CIFAR_STD[:, None, None]

        if self.do_cutout:
            img = Cutout(self.cutout_length)(img)
        return img, label


def cifar10_val_transform(in_data):
    img, label = in_data

    img = (img - CIFAR_MEAN[:, None, None]) / CIFAR_STD[:, None, None]
    return img, label
