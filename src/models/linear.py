import torch.nn as nn
from src.models.base_model import get_resnet18_backbone

class ResNet18_Linear(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.core = get_resnet18_backbone(**kwargs)
        
        # linear head from backbone features
        self.head = nn.Linear(self.core.num_ftrs, kwargs.get('out_features', 1))

    def forward(self, x):
        ftrs = self.core(x)
        return self.head(ftrs)

def get_model(**kwargs):
    return ResNet18_Linear(**kwargs)