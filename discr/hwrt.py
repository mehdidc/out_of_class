import numpy as np
from digits_and_letters import params

from machinepredict.interface import train

# check <https://arxiv.org/pdf/1701.08380.pdf>
train_file = "../data/hwrt.npz"
nb = len(np.load(train_file)['y'])
nb_train = int(0.8 * nb)
nb_valid = int(0.1 * nb)
nb_classes = 369

params['report']['outdir'] = 'hwrt'

params['data'] = {
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
}

if __name__ == '__main__':
    train(params)
