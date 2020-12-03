'''
Reference:
[1] https://github.com/facebookresearch/moco/moco/loader.py
'''
from PIL import ImageFilter
import random
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader

class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_dataloader(data_setting, batch_size):
    import multiprocessing
    num_workers = multiprocessing.cpu_count()
    root = data_setting['root']
    train_transform, test_transform = data_setting['train_transformation'], data_setting['test_transformation']
    if data_setting['dataset'] == 'cifar-10':
        train_loader = DataLoader(CIFAR10(root=root, train=True, transform=TwoCropsTransform(train_transform), download=True), 
                                batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(CIFAR10(root=root, train=False, transform=test_transform, download=True), 
                                batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)  
        memory_loader = DataLoader(CIFAR10(root=root, train=True, transform=test_transform, download=True), 
                                batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        raise "incorrect dataset name"

    return train_loader, test_loader, memory_loader