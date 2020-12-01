import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSiam_MODEL(nn.Module):
    def __init__(self, base_encoder, hidden_units):
        super(SimSiam_MODEL, self).__init__()
        # build up the model
        self.encoder = []
        self.in_feature_dim = 2048
        self.out_feature_dim = 2048
        for name, module in base_encoder.named_children():
            if not isinstance(module, nn.Linear):
                self.encoder.append(module)
            else:
                self.in_feature_dim = module.in_features
        self.encoder = nn.Sequential(*self.encoder)
         # projection mlp
        self.projection_head = nn.Sequential(*[nn.Linear(self.in_feature_dim, 2048, bias=True), 
                  nn.BatchNorm1d(2048),
                  nn.ReLU(inplace=True), 
                  nn.Linear(2048, 2048, bias=True), 
                  nn.BatchNorm1d(2048),
                  nn.ReLU(inplace=True), 
                  nn.Linear(2048, self.out_feature_dim, bias=True),
                  nn.BatchNorm1d(self.out_feature_dim)])
        # prediction mlp
        self.predictor = nn.Sequential(
                  nn.Linear(self.out_feature_dim, 512, bias=True), 
                  nn.BatchNorm1d(512),
                  nn.ReLU(inplace=True), 
                  nn.Linear(512, self.out_feature_dim, bias=True), 
        )

    def forward(self, x):
        x = self.encoder(x)
        representation = torch.flatten(x, start_dim=1)
        projection = self.projection_head(representation)
        prediction = self.predictor(projection)
        return F.normalize(representation, dim=-1), F.normalize(projection, dim=-1), F.normalize(prediction, dim=-1)