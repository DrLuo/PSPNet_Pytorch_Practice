# training and Dataset config


cfg = {
    'layers': 50,  # layers of ResNet, be in [18, 34, 50, 101, 152]
    'bins': (1, 2, 3, 6),  # bins for PSPNet pyramid pooling
    'num_class': 21,
    'max_iter': 30000,
    'pooling': 'AVE',
    'aux_weight': 0.4,
    'ignore_label': 255,
    'zoom_factor': 8,  # zoom factor for final prediction during training, be in [1, 2, 4, 8]
    'rotate_min': -10,
    'rotate_max': 10,
    'resize_min': 0.5,
    'resize_max': 2,
    'train_h': 224,
    'train_w': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}


