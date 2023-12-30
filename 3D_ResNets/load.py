import os

from 2D_to_3D_Weight_Transfer import resnet18_3D, resnet34_3D, resnet50_3D, resnet101_3D, resnet152_3D

# how to import 3D resnet 18
model = resnet18_3D(pre_trained = True, num_classes = 2)

# how to import 3D resnet 34
model = resnet34_3D(pre_trained = True, num_classes = 2)


