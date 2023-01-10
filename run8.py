import mindspore_hub as mshub

model = mshub.load("mindspore/1.9/mobilenetv1_cifar10", pretrained=False)
print(model)