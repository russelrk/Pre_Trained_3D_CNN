from ResNet3D import ResNet
from torchvision import models
def weight_transfer(model3d, model2d):
    # Disable gradient computation for the following operations
    with torch.no_grad():
        # Iterate through corresponding layers in the 3D and 2D models
        for layer3d, layer2d in zip(model3d.children(), model2d.children()):
            
            # Transfer weights for 3D convolutional layers
            if isinstance(layer3d, nn.Conv3d):
                for j in range(7):
                    for i in range(64):
                        # Copy weights from the 2D model to the 3D model
                        layer3d.weight[i, 0, j, :, :].copy_(layer2d.weight[i, 0, :, :])
            
            # Transfer weights for 3D batch normalization layers
            elif isinstance(layer3d, nn.BatchNorm3d):
                # Copy weights from the 2D model to the 3D model
                layer3d.weight.copy_(layer2d.weight)
            
            # Transfer weights for sequential blocks (e.g., ResNet blocks)
            elif isinstance(layer3d, nn.Sequential):
                # Iterate through sub-blocks in the sequential block
                for block3d, block2d in zip(layer3d, layer2d):
                    # Copy weights for the batch normalization layers
                    block3d.bn1.weight.copy_(block2d.bn1.weight)
                    block3d.bn2.weight.copy_(block2d.bn2.weight)
                    for i in range(64):
                        for j in range(64):
                            for k in range(3):
                                # Copy weights for the convolutional layers in the sub-block
                                block3d.conv1.weight[i, j, k, :, :].copy_(block2d.conv1.weight[i, j, :, :])
                                block3d.conv2.weight[i, j, k, :, :].copy_(block2d.conv2.weight[i, j, :, :])
    
    # Return the modified 3D model with transferred weights
    return model3d
    
    
def resnet18_3D(pre_trained=None, num_classes=100):
    """
    Create a 3D ResNet-18 model and optionally transfer pre-trained weights from the 2D ResNet-18.
    
    Args:
        pre_trained (bool): Whether to transfer pre-trained weights from the 2D ResNet-18.
        num_classes (int): Number of output classes.

    Returns:
        model3d (ResNet): 3D ResNet-18 model.
    """
    model3d = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return weight_transfer(model3d, models.resnet18(weights="DEFAULT")) if pre_trained else model3d

def resnet34_3D(pre_trained=None, num_classes=100):
    """
    Create a 3D ResNet-34 model and optionally transfer pre-trained weights from the 2D ResNet-34.
    
    Args:
        pre_trained (bool): Whether to transfer pre-trained weights from the 2D ResNet-34.
        num_classes (int): Number of output classes.

    Returns:
        model3d (ResNet): 3D ResNet-34 model.
    """
    model3d = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return weight_transfer(model3d, models.resnet34(weights="DEFAULT")) if pre_trained else model3d

def resnet50_3D(pre_trained=None, num_classes=100):
    """
    Create a 3D ResNet-50 model and optionally transfer pre-trained weights from the 2D ResNet-50.
    
    Args:
        pre_trained (bool): Whether to transfer pre-trained weights from the 2D ResNet-50.
        num_classes (int): Number of output classes.

    Returns:
        model3d (ResNet): 3D ResNet-50 model.
    """
    model3d = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return weight_transfer(model3d, models.resnet50(weights="DEFAULT")) if pre_trained else model3d

def resnet101_3D(pre_trained=None, num_classes=100):
    """
    Create a 3D ResNet-101 model and optionally transfer pre-trained weights from the 2D ResNet-101.
    
    Args:
        pre_trained (bool): Whether to transfer pre-trained weights from the 2D ResNet-101.
        num_classes (int): Number of output classes.

    Returns:
        model3d (ResNet): 3D ResNet-101 model.
    """
    model3d = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return weight_transfer(model3d, models.resnet101(weights="DEFAULT")) if pre_trained else model3d

def resnet152_3D(pre_trained=None, num_classes=100):
    """
    Create a 3D ResNet-152 model and optionally transfer pre-trained weights from the 2D ResNet-152.
    
    Args:
        pre_trained (bool): Whether to transfer pre-trained weights from the 2D ResNet-152.
        num_classes (int): Number of output classes.

    Returns:
        model3d (ResNet): 3D ResNet-152 model.
    """
    model3d = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    return weight_transfer(model3d, models.resnet152(weights="DEFAULT")) if pre_trained else model3d




