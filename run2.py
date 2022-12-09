import numpy as np

weight = np.array([[-7, 9, 7],
                   [-50, 6, 4],
                   [-10, -5, -7],
                   [-1, -6, -3],
                   [-2, 2, -5]]
                  )
indices = np.array([0, 1, 2, 1, 2])
offsets = np.array([0, 1, 3])
bag_size = np.array([1, 2, 2])
offset2bag = np.array([0, 1, 1, 2, 2])

numIndices = indices.shape[0]
numBags = offsets.shape[0]
featureSize = weight.shape[1]

max_indices = np.zeros((numBags, featureSize), dtype=np.int32)
output = np.zeros((numBags, featureSize), dtype=np.int32)

for i in range(numIndices):
    bag = offset2bag[i]
    word_idx = indices[i]
    for dim in range(featureSize):
        current_item = output[bag, dim]
        weight_item = weight[word_idx, dim]

        is_first_for_bag = (i == 0) or offset2bag[i-1] != bag

        if is_first_for_bag or weight_item > current_item:
            output[bag, dim] = weight_item
            max_indices[bag,dim] = word_idx


print(output)
print(max_indices)
