import tensorflow as tf

indices = [[1, 2, 3]]
values = [2.0]
shape = [4, 3, 6]

start = [1, 2, 2]
size = [1, 1, 1]


y = tf.raw_ops.SparseSlice(
    indices=indices, values=values, shape=shape, start=start, size=size)

print(y)
