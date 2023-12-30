import torch 
import torch.nn as nn
from DenseNet3D import DenseNet3D
from torchvision import models

def weight_transfer_2d_to_3d(model2d, model3d):
    # Disable gradient computation for the following operations
    with torch.no_grad():
        # Iterate through corresponding layers in the 2D and 3D models
        for layer2d, layer3d in zip(model2d.features.children(), model3d.features.children()):
            # Transfer weights for 2D convolutional layers
            if isinstance(layer2d, nn.Conv2d) and isinstance(layer3d, nn.Conv3d):
                # Copy weights from the 2D model to the 3D model
                layer3d.weight[:, 0, :, :, :].copy_(layer2d.weight.unsqueeze(1))
                if layer2d.bias is not None and layer3d.bias is not None:
                    layer3d.bias.copy_(layer2d.bias)

            # Transfer weights for batch normalization layers
            elif isinstance(layer2d, nn.BatchNorm2d) and isinstance(layer3d, nn.BatchNorm3d):
                # Copy weights from the 2D model to the 3D model
                layer3d.weight.copy_(layer2d.weight)
                layer3d.bias.copy_(layer2d.bias)
                layer3d.running_mean.copy_(layer2d.running_mean)
                layer3d.running_var.copy_(layer2d.running_var)

    # Return the modified 3D model with transferred weights
    return model3d


import torch
from torchvision import models
from your_module import DenseNet3D  # Replace 'your_module' with the actual module where you defined the DenseNet3D class

def densenet161_3D(pre_trained=None, num_classes=1000):
    """
    Create a 3D DenseNet-161 model and optionally transfer pre-trained weights from the 2D DenseNet-161.
    
    Args:
        pre_trained (bool): Whether to transfer pre-trained weights from the 2D DenseNet-161.
        num_classes (int): Number of output classes.

    Returns:
        model3d (DenseNet3D): 3D DenseNet-161 model.
    """
    model3d = DenseNet3D(growth_rate=48, block_config=[6, 12, 36, 24], num_init_features=96, num_classes=num_classes)
    
    if pre_trained:
        # Load pre-trained weights from the 2D DenseNet-161 model
        densenet161_2d_model = models.densenet161(pretrained=True)
        
        # Transfer weights from 2D to 3D model
        model3d = weight_transfer_2d_to_3d(densenet161_2d_model, model3d)

    return model3d

def densenet169_3D(pre_trained=None, num_classes=1000):
    """
    Create a 3D DenseNet-169 model and optionally transfer pre-trained weights from the 2D DenseNet-169.
    
    Args:
        pre_trained (bool): Whether to transfer pre-trained weights from the 2D DenseNet-169.
        num_classes (int): Number of output classes.

    Returns:
        model3d (DenseNet3D): 3D DenseNet-169 model.
    """
    model3d = DenseNet3D(growth_rate=32, block_config=[6, 12, 32, 32], num_init_features=64, num_classes=num_classes)
    
    if pre_trained:
        # Load pre-trained weights from the 2D DenseNet-169 model
        densenet169_2d_model = models.densenet169(pretrained=True)
        
        # Transfer weights from 2D to 3D model
        model3d = weight_transfer_2d_to_3d(densenet169_2d_model, model3d)

    return model3d

def densenet201_3D(pre_trained=None, num_classes=1000):
    """
    Create a 3D DenseNet-201 model and optionally transfer pre-trained weights from the 2D DenseNet-201.
    
    Args:
        pre_trained (bool): Whether to transfer pre-trained weights from the 2D DenseNet-201.
        num_classes (int): Number of output classes.

    Returns:
        model3d (DenseNet3D): 3D DenseNet-201 model.
    """
    model3d = DenseNet3D(growth_rate=32, block_config=[6, 12, 48, 32], num_init_features=64, num_classes=num_classes)
    
    if pre_trained:
        # Load pre-trained weights from the 2D DenseNet-201 model
        densenet201_2d_model = models.densenet201(pretrained=True)
        
        # Transfer weights from 2D to 3D model
        model3d = weight_transfer_2d_to_3d(densenet201_2d_model, model3d)

    return model3d


