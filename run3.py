import mindspore
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from mindspore.ops._grad.grad_base import dyn_shape
from mindspore import context
# context.set_context(mode=context.GRAPH_MODE)

@mindspore.jit
def func_bprop(weight, indices, offsets):
    weight, indices, offsets = to_dyn(weight, indices, offsets)
    bag_size = ops.zeros_like(offsets)
    bag_size[:-1] = offsets[1:] - offsets[:-1]
    bag_size[-1] = indices.shape[0] - offsets[-1]

    offset2bag = ops.zeros_like(indices)
    offset2bag = ops.index_fill(offset2bag, 0, offsets[1:], 1)
    offset2bag = ops.cumsum(offset2bag, 0)

    # if ops.is_sequence_shape_unknown(offsets.shape):
    bag_num = dyn_shape(offsets)[0].astype(mindspore.int32)
    # else:
    #     bag_num = ops.scalar_to_tensor(offsets.shape[0])

    maximum_indices = []
    for i in ops.range(ops.scalar_to_tensor(0, mindspore.int32), bag_num, ops.scalar_to_tensor(1, mindspore.int32)):
        idx = ops.masked_select(indices, offset2bag==i)
        weight_bag = ops.gather(weight, idx, 0)
        max_pos = ops.argmax(weight_bag, 0)
        max_pos = ops.gather(idx, max_pos, 0)
        maximum_indices.append(max_pos)
    # maximum_indices = ops.stack(maximum_indices, 0)

    return bag_size, offset2bag, maximum_indices

@mindspore.jit
def to_dyn(weight, indices, offsets):
    offsets = ops.unique(offsets)[0]
    return weight, indices, offsets



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
    print(offsets)

    # ret1, ret2 = foo()
    # print(ret1, ret2)







