import torch.nn as nn
from src.models.base_model import get_resnet18_backbone

class ResNet18_Linear(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.core = get_resnet18_backbone(pretrained)
        
        # linear head from backbone features
        self.head = nn.Linear(self.core.num_ftrs, 1)

    def forward(self, x):
        ftrs = self.core(x)
        return self.head(ftrs)


def get_model(pretrained=True):
    return ResNet18_Linear(pretrained)