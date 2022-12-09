import mindspore
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from mindspore import context
context.set_context(mode=context.GRAPH_MODE)

@mindspore.jit
def func_bprop(weight, indices, offsets):
    bag_size = ops.zeros_like(offsets)
    bag_size[:-1] = offsets[1:] - offsets[:-1]
    bag_size[-1] = indices.shape[0] - offsets[-1]

    offset2bag = ops.zeros_like(indices)
    offset2bag = ops.index_fill(offset2bag, 0, offsets[1:], 1)
    offset2bag = ops.cumsum(offset2bag, 0)

    bag_num = offsets.shape[0]

    maximum_indices = []
    for i in range(bag_num):
        idx = ops.masked_select(indices, offset2bag==i)
        weight_bag = ops.gather(weight, idx, 0)
        max_pos = ops.argmax(weight_bag, 0)
        max_pos = ops.gather(idx, max_pos, 0)
        maximum_indices.append(max_pos)
    maximum_indices = ops.stack(maximum_indices, 0)

    return bag_size, offset2bag, maximum_indices


if __name__ == '__main__':
    weight = Tensor(np.array([[-7, 9, 7],
                       [-50, 6, 4],
                       [-10, -5, -7],
                       [-1, -6, -3],
                       [-2, 2, -5]]
                      ).astype(np.int32))
    indices = Tensor(np.array([0, 1, 2, 1, 2]).astype(np.int32))
    offsets = Tensor(np.array([0, 1, 3]).astype(np.int32))   # include_last_offset=False

    output = func_bprop(weight, indices, offsets)
    print(output)








