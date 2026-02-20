import torch
import torch.nn as nn
from torchvision.models import resnet18

import torch
import torch.nn as nn
from torchvision.models import resnet18

class FashionResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        m = resnet18(weights=None)

        m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()

        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits, feats