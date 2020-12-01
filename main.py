import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import json
from utils import *
from default_settings import get_simclr_setting

# build model
from algorithms import SimCLR
import torchvision.models as models

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='./configs/simclr_cifar10.py', help='The path of config file.')
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for x, target in train_bar:
        x1, x2 = x[0].cuda(), x[1].cuda()
        representation_1, projection_1 = net(x1)
        representation_2, projection_2 = net(x2)

        # [2*B, D]
        out = torch.cat([projection_1, projection_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(projection_1 * projection_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def knn_eval(net, memory_data_loader, test_data_loader, k=200):
    net.eval()

    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            representation, _ = net(data.cuda(non_blocking=True))
            feature_bank.append(representation)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            representation, _ = net(data)
            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(representation, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, num_classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test : Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100

if __name__ == '__main__':
    config = load_config(args.config_file)
    for k, v in config['settings'].items():
        vars()[k] = v
    
    set_seed(seed)
    # data preparation
    data_setting = get_settings(dataset, root)
    train_loader, test_loader, memory_loader = get_dataloader(data_setting, batch_size)
    num_classes = len(test_loader.dataset.classes)

    # model setup and optimizer config
    model = SimCLR(base_encoder=models.__dict__[base_encoder](), hidden_units=hidden_units).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        adjust_learning_rate(optimizer, epoch, lr, epochs)
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = knn_eval(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
    results['best_acc'] = best_acc

    jsonfile = assign_log_name(args.config_file)
    with open(jsonfile, 'w') as out:
        json.dump(results, out, sort_keys=False, indent=4)
    