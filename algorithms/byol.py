import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from usmodels import Byol_Encoder, Predictor, EMA
from tqdm import tqdm
from utils import adjust_learning_rate
from .simclr import SimCLR

class Byol(SimCLR):
    def __init__(self, base_encoder, hidden_units, train_loader, test_loader, memory_loader, train_settings, device):
        SimCLR.__init__(self, base_encoder, hidden_units, train_loader, test_loader, memory_loader, train_settings, device)
        # basic setting
        self.projection_dim = train_settings['projection_dim']
        self.tau = train_settings['tau']
        # online encoder : f + g
        self.model = Byol_Encoder(base_encoder, hidden_units, projection_dim=self.projection_dim).to(self.device)
        # predictor for online encoder
        self.predictor = Predictor(self.projection_dim).to(self.device)
        # target encoder : f + g
        self.target_encoder = Byol_Encoder(base_encoder, hidden_units, projection_dim=self.projection_dim).to(self.device)
        self.set_requires_grad(self.target_encoder, False)
        # target EMA
        self.target_ema_updater = EMA(self.tau)
        self.optimizer = optim.SGD(list(self.model.parameters()) + list(self.predictor.parameters()) + list(self.target_encoder.parameters()), 
                                   lr=self.lr, momentum=train_settings['momentum'], weight_decay=train_settings['weight_decay'])
    
    def L(self, p, z):
        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return 2 - 2 * (p * z).sum(dim=1).mean()

    def set_requires_grad(self, model, val):
        for p in model.parameters():
            p.requires_grad = val

    def update_moving_average(self, ema_updater, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = ema_updater.update_average(old_weight, up_weight)

    def get_representations(self, x):
        with torch.no_grad():
            representation, _ = self.model(x)
        return F.normalize(representation, dim=-1)

    def train(self, epoch, epochs):
        adjust_learning_rate(self.optimizer, epoch, self.lr, epochs)
        self.model.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(self.train_loader)
        for x, target in train_bar:
            batch_size = len(target)
            x1, x2 = x[0].cuda(), x[1].cuda()

            _, online_projection_1 = self.model(x1)
            _, online_projection_2 = self.model(x2)
            online_prediction_1 = self.predictor(online_projection_1)
            online_prediction_2 = self.predictor(online_projection_2)

            with torch.no_grad():
                _, target_projection_1 = self.target_encoder(x1)
                _, target_projection_2 = self.target_encoder(x2)
           
            loss = self.L(online_prediction_1, target_projection_2.detach()) + self.L(online_prediction_2, target_projection_1.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_moving_average(self.target_ema_updater, self.target_encoder, self.model)

            total_num += batch_size
            total_loss += loss.item() * batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        return total_loss / total_num