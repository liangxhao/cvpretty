import torch
import mindspore
import mindspore.ops as ms_ops
import numpy as np
from mindspore import Tensor

input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
output = ms_ops.gather_nd(input_x, indices)
print(output)


a = torch.zeros(2, 3, 4)
b = a.index_put([torch.tensor([1]), torch.tensor([1, 1])], torch.tensor(1.), True)

print(b)
