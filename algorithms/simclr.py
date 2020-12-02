import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from usmodels import SimCLR_MODEL
from tqdm import tqdm
from utils import adjust_learning_rate

class SimCLR(object):
    def __init__(self, base_encoder, hidden_units, train_loader, test_loader, memory_loader, train_settings, device):
        self.device = device
        self.model = SimCLR_MODEL(base_encoder, hidden_units).to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.memory_loader = memory_loader
        self.num_classes = len(test_loader.dataset.classes)
        self.lr = train_settings['lr']
        self.temperature = train_settings['temperature']
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, 
                                    momentum=train_settings['momentum'], weight_decay=train_settings['weight_decay'])

    def get_representations(self, x):
        with torch.no_grad():
            representation, _ = self.model(x)
        return representation

    def train(self, epoch, epochs):
        adjust_learning_rate(self.optimizer, epoch, self.lr, epochs)
        self.model.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(self.train_loader)
        for x, target in train_bar:
            batch_size = len(target)
            x1, x2 = x[0].cuda(), x[1].cuda()
            representation_1, projection_1 = self.model(x1)
            representation_2, projection_2 = self.model(x2)

            # [2*B, D]
            out = torch.cat([projection_1, projection_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

            # compute loss
            pos_sim = torch.exp(torch.sum(projection_1 * projection_2, dim=-1) / self.temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_num += batch_size
            total_loss += loss.item() * batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        return total_loss / total_num

    def test(self, k=200):
        self.model.eval()

        total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
        with torch.no_grad():
            # generate feature bank
            for data, target in tqdm(self.memory_loader, desc='Feature extracting'):
                representation = self.get_representations(data.to(self.device))
                feature_bank.append(representation)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            feature_labels = torch.tensor(self.memory_loader.dataset.targets, device=feature_bank.device)
            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(self.test_loader)
            for data, target in test_bar:
                representation = self.get_representations(data.to(self.device))
                target = target.to(self.device)
                total_num += data.size(0)
                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = torch.mm(representation, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                sim_weight = (sim_weight / self.temperature).exp()

                # counts for each class
                one_hot_label = torch.zeros(data.size(0) * k, self.num_classes, device=sim_labels.device)
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
                # weighted score ---> [B, C]
                pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, self.num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                test_bar.set_description('Test : Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                        .format(total_top1 / total_num * 100, total_top5 / total_num * 100))

        return total_top1 / total_num * 100, total_top5 / total_num * 100