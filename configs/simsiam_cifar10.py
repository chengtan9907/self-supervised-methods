basic_settings = {
    # seed
    'seed': 1,
    # dataset
    'dataset': 'cifar-10',
    'root': '../LabNoise/data',
    'algorithm': 'simsiam',
    # model
    'base_encoder': 'resnet18',
    'hidden_units': 128,
    # train
    'batch_size': 128,
    'epochs': 800,
}
train_settings = {
    'lr': 1e-3,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'temperature': 0.5,
}