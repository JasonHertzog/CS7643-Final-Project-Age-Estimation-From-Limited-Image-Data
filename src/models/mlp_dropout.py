import torch.nn as nn
from src.models.base_model import get_resnet18_backbone

class ResNet18_MLP_Dropout(nn.Module):
    def __init__(self, dropout=0.2, **kwargs):
        super().__init__()
        
        # pass kwargs (pretrained, freeze_backbone, etc.) to backbone
        self.core = get_resnet18_backbone(**kwargs)
        
        # head uses dropout locally
        self.head = nn.Sequential(
            nn.Linear(self.core.num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        ftrs = self.core(x)
        return self.head(ftrs)

def get_model(**kwargs):
    return ResNet18_MLP_Dropout(**kwargs)
