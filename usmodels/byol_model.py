import torch
import torch.nn as nn
import torch.nn.functional as F

class Byol_Encoder(nn.Module):
    def __init__(self, base_encoder, hidden_units, projection_dim):
        super().__init__()
        # build up the model
        self.encoder = []
        self.in_feature_dim = 2048
        for name, module in base_encoder.named_children():
            if not isinstance(module, nn.Linear):
                self.encoder.append(module)
            else:
                self.in_feature_dim = module.in_features
        self.encoder = nn.Sequential(*self.encoder)
         # projection mlp
        self.projection_head = nn.Sequential(*[nn.Linear(self.in_feature_dim, hidden_units, bias=True), 
                  nn.BatchNorm1d(hidden_units),
                  nn.ReLU(inplace=True), 
                  nn.Linear(hidden_units, projection_dim, bias=True),])

    def forward(self, x):
        x = self.encoder(x)
        representation = torch.flatten(x, start_dim=1)
        projection = self.projection_head(representation)
        return representation, projection


class Predictor(nn.Module):
    def __init__(self, projection_dim, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(projection_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_dim)
        )

    def forward(self, x):
        return self.net(x)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new