from 3D_ResNet import ResNet
from torchvision import models

def resnet183D(pre_trained = None, num_classes = 100):
    
    model = ResNet(BasicBlock, [2,2,2,2], num_classes = num_classes)
    
    if pre_trained == True:
        model1 = models.resnet18(weights="DEFAULT")
        
        with torch.no_grad():
            for layer3d, layer2d in zip(model.children(), model1.children()):
                
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
        
    
    return model


def resnet343D(pre_trained = None, num_classes = 100):
    
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)
    
    if pre_trained == True:
        model1 = models.resnet34(weights="DEFAULT")
        
        with torch.no_grad():
            for layer3d, layer2d in zip(model.children(), model1.children()):
                
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
        
    
    return model



def resnet503D(pre_trained = None, num_classes = 100):
    
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes = num_classes)
    
    if pre_trained == True:
        model1 = models.resnet50(weights="DEFAULT")
        
        with torch.no_grad():
            for layer3d, layer2d in zip(model.children(), model1.children()):
                
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
        
    
    return model



def resnet1013D(pre_trained = None, num_classes = 100):
    
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes = num_classes)
    
    if pre_trained == True:
        model1 = models.resnet101(weights="DEFAULT")
        
        with torch.no_grad():
            for layer3d, layer2d in zip(model.children(), model1.children()):
                
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
        
    
    return model


def resnet1013D(pre_trained = None, num_classes = 100):
    
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes = num_classes)
    
    if pre_trained == True:
        model1 = models.resnet152(weights="DEFAULT")
        
        with torch.no_grad():
            for layer3d, layer2d in zip(model.children(), model1.children()):
                
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
        
    
    return model

