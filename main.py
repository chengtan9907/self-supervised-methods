import argparse
import os

import torch
import torch.nn.functional as F

from tqdm import tqdm
import json
from utils import *
from default_settings import get_settings

# build model
from algorithms import SimCLR, SimSiam
import torchvision.models as models

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='./configs/simclr_cifar10.py', help='The path of config file.')
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    config = load_config(args.config_file)
    for k, v in config['basic_settings'].items():
        vars()[k] = v
    
    set_seed(seed)
    # data preparation
    data_setting = get_settings(dataset, root)
    train_loader, test_loader, memory_loader = get_dataloader(data_setting, batch_size)

    # model setup
    if algorithm == 'simclr':
        model = SimCLR(base_encoder=models.__dict__[base_encoder](), hidden_units=hidden_units, 
                   train_loader=train_loader, test_loader=test_loader, memory_loader=memory_loader, 
                   train_settings=config['train_settings'], device=device)
    elif algorithm == 'simsiam':
        model = SimSiam(base_encoder=models.__dict__[base_encoder](), hidden_units=hidden_units, 
                   train_loader=train_loader, test_loader=test_loader, memory_loader=memory_loader, 
                   train_settings=config['train_settings'], device=device)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = model.train(epoch, epochs)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = model.test()
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
    results['best_acc'] = best_acc

    jsonfile = assign_log_name(args.config_file)
    with open(jsonfile, 'w') as out:
        json.dump(results, out, sort_keys=False, indent=4)
    