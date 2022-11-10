import tensorflow as tf
import numpy as np

shape = [6, 3]
values = [[1, 2, 3, 4]]
default_value = 7
row_partition_tensors = [0, 1, 2, 3, 4]
row_partition_types = ['ROW_SPLITS']


y=tf.raw_ops.RaggedTensorToTensor(
    shape=shape,
    values=values,
    default_value=default_value,
    row_partition_tensors=row_partition_tensors,
    row_partition_types=row_partition_types,
)

print(y)