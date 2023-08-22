from 3D_ResNet import ResNet
from torchvision import models

def weight_transfer(model3d, model2d):
    with torch.no_grad():
        for layer3d, layer2d in zip(model3d.children(), model2d.children()):

            if isinstance(layer3d, nn.Conv3d):
                for j in range(7):
                    for i in range(64):
                        layer3d.weight[i, 0, j, :, :].copy_(layer2d.weight[i, 0, :, :])


            elif isinstance(layer3d, nn.BatchNorm3d):
                layer3d.weight.copy_(layer2d.weight)

            elif isinstance(layer3d, nn.Sequential):
                for block3d, block2d in zip(layer3d, layer2d):
                    block3d.bn1.weight.copy_(block2d.bn1.weight)
                    block3d.bn2.weight.copy_(block2d.bn2.weight)
                    for i in range(64):
                        for j in range(64):
                            for k in range(3):
                                block3d.conv1.weight[i, j, k, :, :].copy_(block2d.conv1.weight[i, j, :, :])
                                block3d.conv2.weight[i, j, k, :, :].copy_(block2d.conv2.weight[i, j, :, :])
         
    
    return model3d
    
def resnet18_3D(pre_trained = None, num_classes = 100):
    
    model3d = ResNet(BasicBlock, [2,2,2,2], num_classes = num_classes)
    return weight_transfer(model3d, models.resnet18(weights="DEFAULT")) if pre_trained else model3d


def resnet34_3D(pre_trained = None, num_classes = 100):
    
    model3d = ResNet(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)
    return weight_transfer(model3d, models.resnet34(weights="DEFAULT")) if pre_trained else model3d

def resnet50_3D(pre_trained = None, num_classes = 100):
    
    model3d = ResNet(Bottleneck, [3, 4, 6, 3], num_classes = num_classes)
    return weight_transfer(model3d, models.resnet50(weights="DEFAULT")) if pre_trained else model3d


def resnet101_3D(pre_trained = None, num_classes = 100):
    
    model3d = ResNet(Bottleneck, [3, 4, 23, 3], num_classes = num_classes)
    return weight_transfer(model3d, models.resnet101(weights="DEFAULT")) if pre_trained else model3d

def resnet152_D(pre_trained = None, num_classes = 100):
    
    model3d = ResNet(Bottleneck, [3, 8, 36, 3], num_classes = num_classes)
    return weight_transfer(model3d, models.resnet152(weights="DEFAULT")) if pre_trained else model3d



