import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR_MODEL(nn.Module):
    def __init__(self, base_encoder, hidden_units):
        super(SimCLR_MODEL, self).__init__()
        # build up the model
        self.encoder = []
        self.feature_dim = 2048
        for name, module in base_encoder.named_children():
            if not isinstance(module, nn.Linear):
                self.encoder.append(module)
            else:
                self.feature_dim = module.in_features
        self.encoder = nn.Sequential(*self.encoder)
        self.projection_head = nn.Sequential(
                                nn.Linear(self.feature_dim, hidden_units, bias=False), 
                                nn.BatchNorm1d(hidden_units),
                                nn.ReLU(inplace=True), 
                                nn.Linear(hidden_units, self.feature_dim, bias=True))

    def forward(self, x):
        x = self.encoder(x)
        representation = torch.flatten(x, start_dim=1)
        projection = self.projection_head(representation)
        return F.normalize(representation, dim=-1), F.normalize(projection, dim=-1)