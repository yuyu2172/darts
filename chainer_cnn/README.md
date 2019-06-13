# Chainer reimplementation of DARTS

## Conversion

Download `cifar10_model.pt` first.
Conversion to Chainer weight can be done by

```
python pth2npz.py
```

## Inference

```
python eval.py --gpu 0 --pretrained-model model.npz
```

Converted weight produces the same accuracy as the original PyTorch implementation.

Top 1 error 2.6257991790771484%
(Reference: 2.63%)
