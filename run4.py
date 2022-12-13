import torch
import mindspore
import mindspore.ops as ms_ops
import numpy as np


x = mindspore.Tensor([[1, 2], [3, 4], [5, 6]], mindspore.float32)
v = mindspore.Tensor([[7, 8], [9, 9]], mindspore.float32)
indices = mindspore.Tensor([1, 2], dtype=mindspore.int32)

z = ms_ops.inplace_update(x, v, (1, 2))

print(z)