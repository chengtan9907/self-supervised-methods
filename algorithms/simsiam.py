import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from upmodels import SimSiam_MODEL
from tqdm import tqdm
from utils import adjust_learning_rate
from .simclr import SimCLR

class SimSiam(SimCLR):
    def __init__(self, base_encoder, hidden_units, train_loader, test_loader, memory_loader, train_settings, device):
        SimCLR.__init__(self, base_encoder, hidden_units, train_loader, test_loader, memory_loader, train_settings, device)
        self.model = SimSiam_MODEL(base_encoder, hidden_units).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, 
                                    momentum=train_settings['momentum'], weight_decay=train_settings['weight_decay'])

    def get_representations(self, x):
        with torch.no_grad():
            representation, _, _ = self.model(x)
        return representation

    def D(self, p, z):
        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return  (p * z).sum(dim=1).mean()

    def train(self, epoch, epochs):
        adjust_learning_rate(self.optimizer, epoch, self.lr, epochs)
        self.model.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(self.train_loader)
        for x, target in train_bar:
            batch_size = len(target)
            x1, x2 = x[0].cuda(), x[1].cuda()
            _, projection_1, prediction_1 = self.model(x1)
            _, projection_2, prediction_2 = self.model(x2)

            loss = self.D(prediction_1, projection_2.detach()) / 2 + self.D(prediction_2, projection_1.detach()) / 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_num += batch_size
            total_loss += loss.item() * batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        return total_loss / total_num