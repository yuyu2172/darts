import numpy as np
import chainer
import chainer.functions as F

from ..operations import ReLUConv2DBN
from ..operations import FactorizedReduce
from ..operations import Identity
from ..operations import OPS


class Cell(chainer.Chain):

    def __init__(self, genotype, C_prev_prev, C_prev, C,
                 is_reduction, is_prev_reduction):
        super(Cell, self).__init__()

        with self.init_scope():
            if is_prev_reduction:
                self.preprocess0 = FactorizedReduce(C_prev_prev, C)
            else:
                self.preprocess0 = ReLUConv2DBN(C_prev_prev, C, 1, 1, 0)
            self.preprocess1 = ReLUConv2DBN(C_prev, C, 1, 1, 0)
            self.ops = chainer.ChainList()

        if is_reduction:
            op_names, src_indices = zip(*genotype.reduce)
            concat_indices = genotype.reduce_concat
        else:
            op_names, src_indices = zip(*genotype.normal)
            concat_indices = genotype.normal_concat

        self._compile(C, op_names, src_indices, concat_indices, is_reduction)
        self._is_reduction = is_reduction

    def _compile(
            self, C, op_names, src_indices, concat_indices, is_reduction):
        assert len(op_names) == len(src_indices)
        self._steps = len(op_names) // 2
        self._concat_indices = concat_indices
        self.multiplier = len(concat_indices)
        self._src_indices = src_indices

        for name, index in zip(op_names, src_indices):
            stride = 2 if is_reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self.ops.add_link(op)
    
    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._src_indices[2 * i]]
            h2 = states[self._src_indices[2 * i + 1]]
            op1 = self.ops[2 * i]
            op2 = self.ops[2 * i + 1]

            h1 = op1(h1)
            h2 = op2(h2)

            if chainer.global_config.train and drop_prob > 0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return F.concat([states[i] for i in self._concat_indices], axis=1)


def drop_path(x, drop_prob):
    xp = chainer.cuda.get_array_module(x.data)
    if drop_prob > 0:
        keep_prob = 1 - drop_prob
        mask = xp.random.binomial(
            1, keep_prob, size=(x.shape[0], 1, 1, 1)).astype(np.float32)
        x = x / keep_prob
        x = x * mask
    return x