import chainer.links as L
import chainer.functions as F
import chainer
from chainercv.links import Conv2DBActiv
from chainercv.links import PickableSequentialChain

from .cell import Cell
from .auxiliary_head import AuxiliaryHeadCIFAR


class NetworkCIFAR(chainer.Chain):

    def __init__(
            self, init_C, n_class, n_layer, use_auxiliary,
            drop_path_prob, genotype):
        super(NetworkCIFAR, self).__init__()

        self._n_layer = n_layer
        self._use_auxiliary = use_auxiliary
        self.drop_path_prob = drop_path_prob

        stem_multiplier = 3
        curr_C = stem_multiplier * init_C
        with self.init_scope():
            self.stem = Conv2DBActiv(3, curr_C, pad=1, activ=None)

            prev_prev_C, prev_C, curr_C = curr_C, curr_C, init_C

            self.cells = chainer.ChainList()

            is_prev_reduction = False
            with self.cells.init_scope():
                for i in range(n_layer):
                    if i in [n_layer // 3, 2 * n_layer // 3]:
                        curr_C *= 2
                        is_reduction = True
                    else:
                        is_reduction = False
                    cell = Cell(
                        genotype, prev_prev_C, prev_C, curr_C,
                        is_reduction, is_prev_reduction)
                    is_prev_reduction = is_reduction

                    self.cells.append(cell)
                    prev_prev_C, prev_C = prev_C, cell.multiplier * curr_C

                    if i == 2 * n_layer // 3:
                        auxiliary_in_C = prev_C

            if use_auxiliary:
                self.auxiliary_head = AuxiliaryHeadCIFAR(auxiliary_in_C, n_class)

            self.classifier = L.Linear(prev_C, n_class)

    def forward(self, x):
        aux_logits = None
        s0 = self.stem(x)
        s1 = s0

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._n_layer // 3:
                # TODO: train
            out = F.average_pooling_2d(s1, ksize=s1.shape[2:], pad=0)
            logits = self.classifier(out.reshape((out.shape[0], -1)))
        return logits, aux_logits
