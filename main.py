import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
np.random.seed(1)


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)) or np.any(np.isnan(data_me)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape

loss = 1e-3
##############################################################################
# NCHW
# images = np.random.uniform(low=-100, high=100, size=(244, 25, 51, 174)).astype(np.float32)
images = np.random.random((244, 25, 51, 174)).astype(np.float32)
size = [74, 61]


############################ 1 ###############################################
# tf: align_corners=False, half_pixel_centers=True
# pt: align_corners=False
y_tf = tf.raw_ops.ResizeBilinear(images=images.transpose([0, 2, 3, 1]), size=size, align_corners=False, half_pixel_centers=True).numpy().transpose([0, 3, 1, 2])
y_pt = F.interpolate(torch.from_numpy(images), size, align_corners=False, mode='bilinear').numpy()

allclose_nparray(y_pt, y_tf, loss, loss)

############################ 2 ###############################################
# tf: align_corners=True, half_pixel_centers=False
# pt: align_corners=True
y_tf = tf.raw_ops.ResizeBilinear(images=images.transpose([0, 2, 3, 1]), size=size, align_corners=True, half_pixel_centers=False).numpy().transpose([0, 3, 1, 2])
y_pt = F.interpolate(torch.from_numpy(images), size, align_corners=True, mode='bilinear').numpy()

allclose_nparray(y_pt, y_tf, loss, loss)


############################ 3 ###############################################
# tf: align_corners=False, half_pixel_centers=False
# pt: align_corners=False
y_tf = tf.raw_ops.ResizeBilinear(images=images.transpose([0, 2, 3, 1]), size=size, align_corners=False, half_pixel_centers=False).numpy().transpose([0, 3, 1, 2])
y_pt = F.interpolate(torch.from_numpy(images), size, align_corners=False, mode='bilinear').numpy()

allclose_nparray(y_pt, y_tf, loss, loss)