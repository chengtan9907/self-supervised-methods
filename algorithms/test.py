import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from usmodels import TEST_MODEL
from tqdm import tqdm
from utils import adjust_learning_rate

class TEST(object):
    def __init__(self, base_encoder, hidden_units, train_loader, test_loader, memory_loader, train_settings, device):
        self.device = device
        self.model = TEST_MODEL(feature_dim=512).to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.memory_loader = memory_loader
        self.num_classes = len(test_loader.dataset.classes)
        self.lr = train_settings['lr']
        self.temperature = train_settings['temperature']
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, 
                                    momentum=train_settings['momentum'], weight_decay=train_settings['weight_decay'])

    def D(self, p, z):
        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return (p * z).sum(dim=1).mean()

    def train(self, epoch, epochs):
        adjust_learning_rate(self.optimizer, epoch, self.lr, epochs)
        self.model.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(self.train_loader)
        for x, target in train_bar:
            batch_size = len(target)
            x1, x2 = x[0].cuda(), x[1].cuda()
            _, z1, p1 = self.model(x1)
            _, z2, p2 = self.model(x2)

            loss = self.D(p1, z2.detach()) / 2 + self.D(p2, z1.detach()) / 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_num += batch_size
            total_loss += loss.item() * batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

        return total_loss / total_num

    def test(self):
        self.model.eval()
        k=200
        c=10
        total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
        with torch.no_grad():
            # generate feature bank
            for data, target in tqdm(self.memory_loader, desc='Feature extracting'):
                feature, _, _ = self.model(data.cuda(non_blocking=True))
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            feature_labels = torch.tensor(self.memory_loader.dataset.targets, device=feature_bank.device)
            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(self.test_loader)
            for data, target in test_bar:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature, _, _ = self.model(data)
                feature = F.normalize(feature, dim=1)

                total_num += data.size(0)
                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = torch.mm(feature, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                # sim_weight = (sim_weight / temperature).exp()

                # counts for each class
                one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
                # weighted score ---> [B, C]
                # pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)
                pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c), dim=1)

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                test_bar.set_description('Test Epoch: Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                        .format(total_top1 / total_num * 100, total_top5 / total_num * 100))

        return total_top1 / total_num * 100, total_top5 / total_num * 100