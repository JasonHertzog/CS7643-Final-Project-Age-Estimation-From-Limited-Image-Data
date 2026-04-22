import torch.nn as nn
from torchvision import models

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # ResNet18 backbone
        if pretrained:
            self.model = models.resnet18(weights='IMAGENET1K_V1')
        else:
            self.model = models.resnet18()
        
        # holding feature size (for head)
        self.num_ftrs = self.model.fc.in_ftrs
        
        # needs this replacement so features match expectations
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)
    
def get_resnet18_backbone(pretrained=True):
    return ResNet18Backbone(pretrained)