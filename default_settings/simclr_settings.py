from torchvision import transforms

DATASETS_INFO = {
    'cifar-10': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
        'image_size': 32,
        'batch_size': 128,
        'epochs': 100
    },

}

def get_simclr_setting(dataset:str = None, root:str = None):
    dataset_info = DATASETS_INFO[dataset]
    data_setting = {
        'dataset': dataset,
        'root': root,
        'train_transformation': transforms.Compose([
            transforms.RandomResizedCrop(dataset_info['image_size']),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # Todo -> Gaussian Blur
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(dataset_info['mean'], dataset_info['std'])]),
        'test_transformation': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
    }
    return data_setting