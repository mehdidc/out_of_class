import numpy as np

train_file = "../data/digits_and_letters.npz"
nb_train = 90000
nb_valid = 5000
nb_classes = 10 + 26
basic_params = {
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

def digits_and_letters():
    return basic_params

def digits_and_letters_balanced():
    params = basic_params.copy()
    train_file = '../data/digits_and_letters_train.npz'
    test_file = '../data/digits_and_letters_test.npz'
    test = np.load(test_file)
    nb = len(test['X'])
    nb_valid = int(nb * 0.3)
    nb_test = nb - nb_valid
    print(nb_valid, nb_test)
    params['report']['outdir'] = 'digits_and_letters_balanced'
    params['data'] = {
        'train': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": train_file, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},

            ]
        },
        'valid': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": test_file, "start": 0, "nb": nb_valid, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},
            ]
        },
        'test': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": test_file, "start": nb_valid, "nb": nb_test, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},
            ]
        },
        'transformers':[
        ]
    }
    return params



def letters():
    params = basic_params.copy()
    train_file = "../data/letters.npz"
    nb = len(np.load(train_file)['y'])
    nb_train = int(0.8 * nb)
    nb_valid = int(0.1 * nb)
    nb_test = nb - nb_train - nb_valid
    nb_classes = 26
    params['report']['outdir'] = 'letters'
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
                {"name": "load_numpy", "params": {"filename": train_file, "start": nb_train + nb_valid, "nb": nb_test, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},
            ]
        },
        'transformers':[
        ]
    }
    return params

def digits():
    params = basic_params.copy()
    train_file = "../data/digits.npz"
    test_file = "../data/digits_test.npz"
    nb = 60000
    nb_train = int(0.9 * nb)
    nb_valid = nb - nb_train
    nb_classes = 10
    params['report']['outdir'] = 'digits'
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
                {"name": "load_numpy", "params": {"filename": test_file, "start": 0, "nb": 10000, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},
            ]
        },
        'transformers':[
        ]
    }
    return params


def hwrt():
    # check <https://arxiv.org/pdf/1701.08380.pdf>
    params = basic_params.copy()
    train_file = "../data/hwrt_padded.npz"
    data = np.load(train_file)
    nb = len(data['y'])
    nb_classes = len(set(data['y']))
    nb_train = int(0.9 * nb)
    nb_valid = int(0.05 * nb)

    params['report']['outdir'] = 'hwrt'
    print(nb, nb_train, nb_valid)

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
    return params


def digits_and_hwrt():
    params = basic_params.copy()
    train_file = "../data/digits_and_hwrt.npz"
    nb = len(np.load(train_file)['y'])
    nb_train = int(0.9 * nb)
    nb_valid = int(0.05 * nb)
    nb_classes = 343 + 10
    params['report']['outdir'] = 'digits_and_hwrt'

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
    return params

def real_vs_fake():
    params = basic_params.copy()
    train_file = "../data/real_vs_fake.npz"
    nb = len(np.load(train_file)['y'])
    nb_train = int(0.8 * nb)
    nb_valid = int(0.1 * nb)
    nb_test = nb - nb_train - nb_valid
    nb_classes = 2
    params['report']['outdir'] = 'real_vs_fake'
    params['data'] = {
        'train': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": train_file, "start": 0, "nb": nb_train, "cols": ["X", "y"]}},
                #{"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},

            ]
        },
        'valid': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": train_file, "start": nb_train, "nb": nb_valid, "cols": ["X", "y"]}},
                #{"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},
            ]
        },
        'test': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": train_file, "start": nb_train + nb_valid, "nb": nb_test, "cols": ["X", "y"]}},
                #{"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},
            ]
        },
        'transformers':[
        ]
    }
    return params


def emnist():
    params = basic_params.copy()
    train_file = '../data/emnist_train.npz'
    test_file = '../data/emnist_test.npz'
    test = np.load(test_file)
    nb_classes = len(set(test['y']))
    nb = len(test['X'])
    nb_valid = int(nb * 0.1)
    nb_test = nb - nb_valid
    params['report']['outdir'] = 'emnist'
    params['data'] = {
        'train': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": train_file, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},

            ]
        },
        'valid': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": test_file, "start": 0, "nb": nb_valid, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},
            ]
        },
        'test': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": test_file, "start": nb_valid, "nb": nb_test, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255.}},
                {"name": "onehot", "params": {"nb_classes": nb_classes}},
            ]
        },
        'transformers':[
        ]
    }
    return params

def quickdraw():
    params = basic_params.copy()
    train_file = "../data/quickdraw.npz"
    nb = len(np.load(train_file)['y'])
    nb_train = int(0.9 * nb)
    nb_valid = int(0.05 * nb)
    nb_classes = 88
    params['report']['outdir'] = 'quickdraw'

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
    return params

