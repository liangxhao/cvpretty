import torch
import numpy as np
import mindspore as ms
import mindspore.ops as ms_ops


np.random.seed(1)
torch.random.manual_seed(1)


x1 = np.random.randint(-128, 128, (5, 6, 7, 8)).astype(np.float32)
x2 = np.random.randint(-128, 128, 1).astype(np.float32)
indices = [[1]]

###############################################################
x1_pt = torch.tensor(x1, requires_grad=True)
x2_pt = torch.tensor(x2, requires_grad=True)
indices_pt = [torch.tensor(x).long() for x in indices]
accumulate = False
result = torch.index_put(x1_pt, indices_pt, x2_pt, accumulate=accumulate)
grad_pt = torch.randint(-128, 128, size=result.shape, dtype=result.dtype)

result.backward(grad_pt)
print(x2_pt.grad)


###############################################################
grad_ms = ms.Tensor(grad_pt.numpy())
indices_ms = [ms.Tensor(x, dtype=ms.int32) for x in indices]
x2_ms = ms.Tensor(x2)

maxsize = max(x.shape[0] for x in indices_ms)
align_indices_ms = [ms_ops.tile(x, (maxsize, )) if x.shape[0] == 1 else x for x in indices_ms]
align_indices_ms = ms_ops.stack(align_indices_ms).T
values_grad = ms_ops.gather_nd(grad_ms, align_indices_ms)
if x2_ms.shape[0] == 1:
    values_grad = values_grad.sum().reshape(1)

print(values_grad)

