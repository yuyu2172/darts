import chainer
import chainer.functions as F
import chainer.links as L


class Cell(chainer.Chain):

    def __init__(self, genotype, C_prev_prev, C_prev, C,
                 is_reduction, is_prev_reduction):
        super(Cell, self).__init__()

        with self.init_scope():
            self.preprocess0 = ReLUConv2DBN(C_prev_prev, C, 1, 1, 0)
            self.preprocess1 = ReLUConv2DBN(C_prev, C, 1, 1, 0)
            self.ops = chainer.ChainList()

        if is_reduction:
            raise ValueError
        else:
            op_names, src_indices = zip(*genotype.normal)
            concat_indices = genotype.normal_concat

        self._compile(C, op_names, src_indices, concat_indices, is_reduction)

    def _compile(
            self, C, op_names, src_indices, concat_indices, is_reduction):
        assert len(op_names) == len(src_indices)
        self._steps = len(op_names) // 2
        self._concat_indices = concat_indices
        self.multiplier = len(concat_indices)

        for name, index in zip(op_names, src_indices):
            stride = 2 if is_reduction and index < 2 else 1
            op = None



if __name__ == '__main__':
    import sys
    l = ReLUConv2DBN(3, 3, 1, 1, 0)
    sys.path.append('../cnn')

    from genotypes import DARTS

    cell = Cell(3, 3, 3, DARTS, False, False)


