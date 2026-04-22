import torch.nn as nn
from src.models.base_model import get_resnet18_backbone

class ResNet18_MLP_Dropout(nn.Module):
    def __init__(self, pretrained=True, dropout=0.2):
        super().__init__()
        
        self.core = get_resnet18_backbone(pretrained)
        
        # sequential head --> linear, ReLU activation, dropout, linear
        self.head = nn.Sequential(
            nn.Linear(self.core.num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        ftrs = self.core(x)
        return self.head(ftrs)

def get_model(pretrained=True, dropout=0.2):
    return ResNet18_MLP_Dropout(pretrained, dropout)