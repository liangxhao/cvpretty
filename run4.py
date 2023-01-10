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
    weight = Tensor(np.array([[-0.5649585,0.61113364,0.5611087,1.6866108,-0.09905473],
                                [ 0.6700379,0.5393704,0.6244112,-0.36923242,-1.040878  ],
                                [-1.022919,-1.5475533,0.7926442,1.7364109,-0.2880044 ],
                                [-0.19605514,1.11115,-0.8076033,-0.8941912,0.634594  ],
                                [ 1.1184089,-0.6195012,-2.009596,-0.34388176,1.3961227 ]]
                                ).astype(np.int32))
    indices = Tensor(np.array([2, 4, 3, 3, 2, 3, 0, 4, 2]).astype(np.int32))
    offsets = Tensor(np.array([0, 6, 7, 7, 8, 8]).astype(np.int32))   # include_last_offset=False

    output = func_bprop(weight, indices, offsets)
    print(output)
