import torch
import mindspore
import mindspore.ops as ms_ops
import numpy as np


np.random.seed(1)

x = np.random.random((3, 10))
p = 2
maxnorm=0.5

x_pt = torch.from_numpy(x)
indices_pt = torch.Tensor([0, 1, 2]).long()
y_pt = torch.embedding_renorm_(x_pt, indices_pt, maxnorm, p)
y_pt = y_pt.numpy()


x_ms = mindspore.Tensor(x)
y_ms = ms_ops.renorm(x_ms, p, 1, maxnorm=maxnorm)
y_ms = y_ms.asnumpy()


assert np.allclose(y_ms, y_pt)
