import torch.nn as nn
from src.models.base_model import get_resnet18_backbone

class ResNet18_MLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        # pass all kwargs down (including pretrained, freeze_backbone, etc.)
        self.core = get_resnet18_backbone(**kwargs)
        
        # head with bottleneck of 256 features
        self.head = nn.Sequential(
            nn.Linear(self.core.num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        ftrs = self.core(x)
        return self.head(ftrs)


def get_model(**kwargs):
    return ResNet18_MLP(**kwargs)