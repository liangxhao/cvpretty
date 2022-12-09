import tensorflow as tf
import numpy as np
x1 = list([i for i in range(10000)])
x1 = tf.convert_to_tensor(x1, dtype=tf.int32)

x2 = tf.convert_to_tensor([20988], dtype=tf.int32)

x3 = list([i for i in range(10000)])
x3 = tf.convert_to_tensor(x3, dtype=tf.float32)

# y = tf.gather_nd(x1, x2)
# y = tf.raw_ops.SparseToDense(sparse_indices=x1, output_shape=x2, sparse_values=x3, default_value=0)

with tf.GradientTape() as g:
    g.watch(x1)
    y = tf.raw_ops.SparseToDense(sparse_indices=x1, output_shape=x2, sparse_values=x3, default_value=0)
    dx = g.gradient(y, x3)

print(dx)

