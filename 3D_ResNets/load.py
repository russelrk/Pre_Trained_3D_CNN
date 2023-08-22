import os

from 2D_to_3D_Weight_Transfer import resnet183D, resnet343D, resnet503D, resnet1013D, resnet1523D

# how to import 3D resnet 18
model = resnet183D(pre_trained = True, num_classes = 2)

# how to import 3D resnet 34
model = resnet343D(pre_trained = True, num_classes = 2)


