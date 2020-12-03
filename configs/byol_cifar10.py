basic_settings = {
    # seed
    'seed': 1,
    # dataset
    'dataset': 'cifar-10',
    'root': '../LabNoise/data',
    'algorithm': 'byol',
    # model
    'base_encoder': 'resnet18',
    'hidden_units': 4096,
    # train
    'batch_size': 256,
    'epochs': 800,
}
train_settings = {
    'lr': 0.03,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'temperature': 0.5,
    'tau': 0.996,
    'projection_dim': 256,
}