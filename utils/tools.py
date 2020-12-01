import math
from config import Config
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import os.path as osp

# cosine decay learning rate schedule
def adjust_learning_rate(optimizer, epoch, lr, epochs):
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# set random seed for reproduction
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

# load and print the config file
def load_config(filename:str = None):
    print('loading config from ' + filename + ' ...')
    configfile = Config(filename=filename)
    config = configfile._cfg_dict
    print('---------- params info: ----------')
    for k, v in config.items():
        print(k, ' : ', v)
    print('---------------------------------')
    return config

def assign_log_name(config_name: str = None):
    filename = config_name.split('/')[-1].replace('.py', '.json')
    if osp.exists('./logs') is False:
        os.mkdir('./logs')
    return osp.join('./logs', filename)