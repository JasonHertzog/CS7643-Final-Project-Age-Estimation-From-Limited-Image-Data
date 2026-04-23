import torch.nn as nn
from torchvision import models

class ResNet18Backbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pretrained = kwargs.get('pretrained', True)
        freeze_backbone = kwargs.get('freeze_backbone', False)

        # ResNet18 backbone
        if pretrained:
            self.model = models.resnet18(weights='IMAGENET1K_V1')
        else:
            self.model = models.resnet18()

        # holding feature size (for head)
        self.num_ftrs = self.model.fc.in_features

        # needs this replacement so features match expectations
        # Now model(x) returns the 512-dimensional feature vector directly
        self.model.fc = nn.Identity()

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

def get_resnet18_backbone(**kwargs):
    return ResNet18Backbone(**kwargs)
