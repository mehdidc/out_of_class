import os

basic_train_params = {
    'family': 'autoencoder',
    'input_col': 'X',
    'output_col': 'X',
    'model': {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [64, 64, 64],
            'encode_filter_sizes': [5, 5, 5],
            'encode_activations': ['relu', 'relu', 'relu'],
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_channel', 'params': {'stride': 4}},
            ],
            'decode_nb_filters': [64, 64],
            'decode_filter_sizes': [5, 5],
            'decode_activations': ['relu', 'relu'],
            'output_filter_size': 5,
            'output_activation': 'sigmoid'
         }
    },
    'data': {
        'train': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": "../data/digits.npz"}},
                {"name": "divide_by", "params": {"value": 255.}},
            ]
        },
        'transformers':[
        ]
    },
    'report':{
        'outdir': 'mnist',
        'checkpoint': {
            'loss': 'train_mean_squared_error',
            'save_best_only': True
        },
        'metrics': ['mean_squared_error'],
        'callbacks': ['image_reconstruction', 'image_features'],
    },
    'optim':{
        'algo': {
            'name': 'adadelta',
            'params': {'lr': 0.1}
        },
        'lr_schedule':{
            'name': 'constant',
            'params': {}
        },
        'early_stopping':{
            'name': 'none',
            'params': {
            }
        },
        'max_nb_epochs': 100,
        'batch_size': 64,
        'pred_batch_size': 128,
        "loss": "squared_error",
        'budget_secs': 86400,
        'seed': 42
    },
}

basic_generate_params = {
    'model':{
        'folder': 'mnist'
    },
    'method':{
        'name': 'iterative_refinement',
        'params': {
            'batch_size': 128,
            'nb_samples': 10000,
            'nb_iter': 100,
            'binarize':{
                'name': 'none',
                'params': {
                }
            },
            'noise':{
                'name': 'none',
                'params': {
                }
            },
            'stop_if_unchanged': False,
            'seed': 42,
        },
        'save_folder': 'mnist/gen',
    }
}

base_folder = 'results'

def mnist():
    t = basic_train_params.copy()
    g = basic_generate_params.copy()
    t, g = _set_folder(t, g, 'mnist')
    return t, g


def mnist_without_sparsity():
    t, g = mnist()
    t, g = _set_folder(t, g, 'mnist_without_sparsity')
    t['model']['params']['code_activations']  = []
    return t, g

def cifar():
    t, g = mnist()
    t, g = _set_folder(t, g, 'cifar')
    dataset = '../data/cifar10.npz'
    nb_centers = 50
    t['model']['params'] = {
        'stride': 1,
        'encode_nb_filters': [64, 64, 64],
        'encode_filter_sizes': [5, 5, 5],
        'encode_activations': ['relu', 'relu', 'relu'],
        'code_activations': [
            {'name': 'winner_take_all_spatial', 'params': {}},
            {'name': 'winner_take_all_channel', 'params': {'stride': 4}},
        ],
        'decode_nb_filters': [64, 64, 64, 64],
        'decode_filter_sizes': [3, 3, 3, 3],
        'decode_activations': ['relu', 'relu', 'relu', 'relu'],
        'output_filter_size': 5,
        'output_activation': {'name': 'axis_softmax', 'params': {'axis': 1}},
    }
    t['data']  = {
        'train': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": dataset}},
                {"name": "divide_by", "params": {"value": 255.}},
            ]
        },
        'transformers':[
            {'name': 'ColorDiscretizer', 'params': {'nb_centers': nb_centers}}
        ]
    }
    t['loss'] = {'name': 'axis_categorical_crossentropy', 'params' : {'axis': 1}}
    return t, g

def svhn():
    t, g = cifar()
    t, g = _set_folder(t, g, 'svhn')
    
    dataset = '../data/svhn.npz'
    t['data']['train'] = {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": dataset}},
                {"name": "divide_by", "params": {"value": 255.}},
            ]
    }
    return t, g

def _set_folder(t, g, folder):
    folder = os.path.join(base_folder, folder)
    t['report']['outdir'] = folder
    g['model']['folder'] = folder
    g['method']['params']['save_folder'] = '{}/gen'.format(folder)
    return t, g
