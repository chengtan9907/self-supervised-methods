import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class TEST_MODEL(nn.Module):
    def __init__(self, feature_dim=128):
        super(TEST_MODEL, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection mlp
        self.projection = nn.Sequential(*[nn.Linear(512, 2048, bias=True), 
                  nn.BatchNorm1d(2048),
                  nn.ReLU(inplace=True), 
                  nn.Linear(2048, 2048, bias=True), 
                  nn.BatchNorm1d(2048),
                  nn.ReLU(inplace=True), 
                  nn.Linear(2048, feature_dim, bias=True),
                  nn.BatchNorm1d(feature_dim)])
        # prediction mlp
        self.h = nn.Sequential(
                  nn.Linear(feature_dim, 512, bias=True), 
                  nn.BatchNorm1d(512),
                  nn.ReLU(inplace=True), 
                  nn.Linear(512, feature_dim, bias=True), 
        )

    def forward(self, x):
        x = self.f(x)        
        x = torch.flatten(x, start_dim=1)
        z = self.projection(x)
        p = self.h(z)
        return x, z, p
