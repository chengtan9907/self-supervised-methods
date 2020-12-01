settings = {
    # seed
    'seed': 1,
    # dataset
    'dataset': 'cifar-10',
    'root': '../LabNoise/data',
    'algorithm': 'simclr',
    # model
    'base_encoder': 'resnet18',
    # train 
    'knn_eval': True,
    'hidden_units': 128,
    'batch_size': 128,
    'epochs': 800,
    'lr': 1e-3,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'temperature': 0.5,
}