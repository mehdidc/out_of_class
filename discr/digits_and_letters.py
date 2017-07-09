import numpy as np

from machinepredict.interface import train

train_file = "../data/digits_and_letters.npz"
nb_train = 90000
nb_valid = 5000
nb_classes = 10 + 26

params = {
    'input_col': 'X',
    'output_col': 'y',
    'model': {
        'name': 'lenet',
        'params':{
            "nb_filters": [64, 128, 256],
            "dropout": 0,
            "fc_dropout": 0.5,
            "batch_norm": False,
            "fc": [1024],
            "size_filters": 3,
            "activation": "prelu",
            "output_activation": "softmax",
         }
    },
    'data': {
        'train': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": train_file, "start": 0, "nb": nb_train, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},
            ]
        },
        'valid': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": train_file, "start": nb_train, "nb": nb_valid, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},
            ]
        },
        'test': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": train_file, "start": nb_train + nb_valid, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},
            ]
        },
        'transformers':[
        ]
    },
    'report':{
        'outdir': 'digits_and_letters',
        'checkpoint': {
            'loss': 'valid_accuracy',
            'save_best_only': True
        },
        'metrics': ['accuracy'],
        'callbacks': [],
    },
    'optim':{
        'algo': {
            'name': 'adam',
            'params': {'lr': 1e-3}
        },
        'lr_schedule':{
            'name': 'decrease_when_stop_improving',
            'params': {
                'patience': 20,
                'loss': 'valid_accuracy',
                'shrink_factor': 2.,
            }
        },
        'early_stopping':{
            'name': 'none',
            'params': {
            }
        },
        'max_nb_epochs': 300,
        'batch_size': 128,
        'pred_batch_size': 128,
        "loss": "categorical_crossentropy",
        'budget_secs': 86400,
        'seed': 42
    },
}

if __name__ == '__main__':
    train(params)
