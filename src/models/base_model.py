import torch
import torch.nn as nn
from torchvision import models

class BaseAgeEstimationModel(nn.Module):
    def __init__(self, pretrained):
        super(BaseAgeEstimationModel, self).__init__()
        # Using ResNet-18 as the baseline architecture
        if pretrained:
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        else:
            self.backbone = models.resnet18()
        
        # Replace the final fully connected layer for regression (1 output for Age)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.backbone(x)

def get_model(pretrained=True):
    return BaseAgeEstimationModel(pretrained)
