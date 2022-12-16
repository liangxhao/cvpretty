import mindspore
import numpy as np
import torch
import mindspore as ms


def get_pt_dx2(x1_pt, x2_pt, indices_pt, grad_pt):
    x2_pt.requires_grad_(True)
    accumulate = False
    result = torch.index_put(x1_pt, indices_pt, x2_pt, accumulate=accumulate)
    result.backward(grad_pt)
    return x2_pt.grad.detach()


@mindspore.jit
def get_ms_dx2(x1_ms, x2_ms, indices_ms, grad_ms):
    maxsize = max(x.shape[0] for x in indices_ms)
    align_indices_ms = [ms.ops.tile(x, (maxsize,)) if x.shape[0] == 1 else x for x in indices_ms]
    align_indices_ms = ms.ops.stack(align_indices_ms).T
    values_grad = ms.ops.gather_nd(grad_ms, align_indices_ms)
    if x2_ms.shape[0] == 1:
        values_grad = values_grad.sum().reshape(1)
    return values_grad

if __name__ == '__main__':
    x1 = np.random.randint(-128, 128, (5, 6, 7, 8)).astype(np.float32)
    x2 = np.random.randint(-128, 128, 1).astype(np.float32)
    indices = [[2]]
    # x2 = np.random.randint(-128, 128, 2).astype(np.float32)
    # indices = [[2], [1,2], [2, 3], [3, 4]]
    grad = np.random.randint(-128, 128, x1.shape).astype(x1.dtype)

    # pytorch
    x1_pt = torch.tensor(x1)
    x2_pt = torch.tensor(x2)
    indices_pt = [torch.tensor(x).long() for x in indices]
    grad_pt = torch.tensor(grad)
    dx2_pt = get_pt_dx2(x1_pt, x2_pt, indices_pt, grad_pt).numpy()

    # mindspore
    x1_ms = ms.Tensor(x1)
    x2_ms = ms.Tensor(x2)
    indices_ms = [ms.Tensor(x, dtype=ms.int32) for x in indices]
    grad_ms = ms.Tensor(grad)
    dx2_ms = get_ms_dx2(x1_ms, x2_ms, indices_ms, grad_ms).asnumpy()

    print("pytorch: ", dx2_pt)
    print("mindspore: ", dx2_ms)

    assert np.allclose(dx2_ms, dx2_pt)


