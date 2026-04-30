import torch.nn as nn
from src.models.base_model import get_resnet18_backbone

class ResNet18_MLP(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.core = get_resnet18_backbone()
        
        # sequential head --> linear, ReLU activation, linear
        self.head = nn.Sequential(
            nn.Linear(self.core.num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    # forward pass through backbone + head
    def forward(self, x):
        ftrs = self.core(x)
        return self.head(ftrs)


def get_model(pretrained=True):
    return ResNet18_MLP(pretrained)