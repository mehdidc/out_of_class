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
            'nb_samples': 100,
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

def mnist_vertebrate():
    t, g = mnist()
    t, g = _set_folder(t, g, 'mnist_vertebrate')
    t['model'] = {
        'name': 'vertebrate',
        'params': {
            'stride': 1,
            'encode_nb_filters': [64, 64],
            'encode_filter_sizes': [5, 5],
            'encode_activations': ['relu', 'relu'],
            'code_activations': [
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
            ],
            'decode': [
                {
                    'nb_filters': [],
                    'filter_sizes': [],
                    'activations': [],
                    'output_filter_size': 5,
                },
                {
                    'nb_filters': [],
                    'filter_sizes': [],
                    'activations': [],
                    'output_filter_size': 9,
                },
            ],
            'output_activation': 'sigmoid',
        },
    }
    return t, g


def olivetti():
    t, g = mnist()
    t, g = _set_folder(t, g, 'olivetti')
    t['optim']['algo']= {
        'name': 'adam',
        'params': {'lr': 1e-3}
    }
    t['report']['metrics'] = ['precision']
    t['report']['checkpoint'] = {
        'loss': 'train_precision',
        'save_best_only': True,
        'mode': 'max',
    }
    t['model']['params'] = {
        'stride': 1,
        'encode_nb_filters': [64, 64, 64],
        'encode_filter_sizes': [5, 5, 5],
        'encode_activations': ['relu', 'relu', 'relu'],
        'code_activations': [
            {'name': 'winner_take_all_spatial', 'params': {}},
            #{'name': 'winner_take_all_channel', 'params': {'stride': 4}},
        ],
        'decode_nb_filters': [64, 64],
        'decode_filter_sizes': [5, 5],
        'decode_activations': ['relu', 'relu'],
        'output_filter_size': 5,
        #'output_activation': 'sigmoid',
        'output_activation': {'name': 'axis_softmax', 'params': {'axis': 1}},
    }
    t['optim']['loss'] = {'name': 'axis_categorical_crossentropy', 'params' : {'axis': 1}}
    t['data'] = {
        'train': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": '../data/olivetti.npz'}},
                {"name": "divide_by", "params": {"value": 255.}},
            ],
        },
        'transformers':[
            {'name': 'ColorDiscretizer', 'params': {'nb_centers': 8}}
        ]
    }
    t["optim"]["max_nb_epochs"] = 1000
    return t, g


def mnist_dense():
    t, g = mnist()
    t,g = _set_folder(t, g, 'mnist_dense')
    t["optim"]["max_nb_epochs"] = 1000
    """
    t['model'] = {
        'name': 'fully_connected',
        'params':{
            'nb_hidden_units': [400, 900, 100, 200, 800],
            #'nb_hidden_units': [400, 900, 100, 200, 200],
            #'activations': ['relu'] * 5 + [{'name': 'ZeroMasking', 'params': {'proba': 0.15125}}],
            #'activations': ['relu'] * 5 + [{'name': 'ksparse', 'params': {'proba': 0.15125}}],
            'activations': ['relu'] * 5 + [{'name': 'ksparse', 'params': {'zero_ratio': 0.74}}],
            #'activations': ['relu'] * 5,
            'output_activation': 'sigmoid',
         }
    }
    """
    t['model'] = [
        {
            'name': 'noise',
            'params': {
                'type': 'salt_and_pepper',
                'params': {
                    'proba': 0.14
                }
            }
        },
        {
            'name': 'fully_connected',
            'params':{
                'nb_hidden_units': [400, 900, 100, 200, 800],
                #'nb_hidden_units': [400, 900, 100, 200, 200],
                #'activations': ['relu'] * 5 + [{'name': 'ZeroMasking', 'params': {'proba': 0.15125}}],
                #'activations': ['relu'] * 5 + [{'name': 'ksparse', 'params': {'proba': 0.15125}}],
                'activations': ['relu'] * 5 + [{'name': 'ksparse', 'params': {'zero_ratio': 0.74}}],
                #'activations': ['relu'] * 5,
                'output_activation': 'sigmoid',
             }
        }
    ]
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 100,
        'nb_iter': 100,
        'binarize':{
            'name': 'none',
            'params': {
                'one_ratio': 0.15,
                'is_moving': True,
            }
        },
        'noise':{
            'name': 'none',
            'params': {
            }
        },
        'stop_if_unchanged': False,
        'seed': 42,
    }
    return t, g


def mnist_without_sparsity():
    t, g = mnist()
    t, g = _set_folder(t, g, 'mnist_without_sparsity')
    t['model']['params']['code_activations']  = []
    return t, g


def cifar():
    t, g = mnist()
    t, g = _set_folder(t, g, 'cifar')
    t['optim']['max_nb_epochs'] = 1000
    dataset = '../data/cifar10.npz'
    nb_centers = 50
    t['optim']['algo']= {
        'name': 'adam',
        'params': {'lr': 1e-3}
    }
    t['report']['metrics'] = ['precision']
    t['report']['checkpoint'] = {
        'loss': 'train_precision',
        'save_best_only': True,
        'mode': 'max',
    }
    t['model']['params'] = {
        'stride': 1,
        'encode_nb_filters': [64, 128, 256],
        'encode_filter_sizes': [5, 5, 5],
        'encode_activations': ['relu', 'relu', 'relu'],
        'code_activations': [
            {'name': 'winner_take_all_spatial', 'params': {}},
            #{'name': 'winner_take_all_channel', 'params': {'stride': 4}},
        ],
        'decode_nb_filters': [256, 256],
        'decode_filter_sizes': [5, 5],
        'decode_activations': ['relu', 'relu'],
        'output_filter_size': 5,
        'output_activation': {'name': 'axis_softmax', 'params': {'axis': 1}},
        #'output_activation': 'sigmoid',
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
    t['optim']['loss'] = {'name': 'axis_categorical_crossentropy', 'params' : {'axis': 1}}
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 100,
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
    }
    return t, g


def shoes():
    t, g = cifar()
    t, g = _set_folder(t, g, 'shoes')
    dataset = '../data/shoes.npz'
    t['data']['train'] = {
        'pipeline':[
            {"name": "load_numpy", "params": {"filename": dataset}},
            {"name": "divide_by", "params": {"value": 255.}},
        ]
    }
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

def celeba():
    t, g = cifar()
    t, g = _set_folder(t, g, 'celeba')
    dataset = '../data/celeba.npz'
    t['data']['train'] = {
        'pipeline':[
            {"name": "load_numpy", "params": {"filename": dataset}},
            {"name": "divide_by", "params": {"value": 255.}},
        ]
    }
    return t, g

def celeba_constrained():
    t, g = celeba()
    t, g = _set_folder(t, g, 'celeba_constrained')
    t['model']['params'] = {
        'stride': 2,
        'encode_nb_filters': [64, 128, 256],
        'encode_filter_sizes': [5, 5, 5],
        'encode_activations': ['relu', 'relu', 'relu'],
        'code_activations': [
            {'name': 'winner_take_all_spatial', 'params': {}},
            #{'name': 'winner_take_all_channel', 'params': {'stride': 4}},
        ],
        'decode_nb_filters': [256, 256],
        'decode_filter_sizes': [5, 5],
        'decode_activations': ['relu', 'relu'],
        'output_filter_size': 5,
        'output_activation': {'name': 'axis_softmax', 'params': {'axis': 1}},
        #'output_activation': 'sigmoid',
    }
    #t['data']['train']['pipeline'][0]['params']["nb"] = 10
    return t, g

def hwrt():
    t, g = mnist()
    t, g = _set_folder(t, g, 'hwrt')
    dataset = '../data/hwrt.npz'
    t['data']['train'] = {
        'pipeline':[
            {"name": "load_numpy", "params": {"filename": dataset}},
            {"name": "divide_by", "params": {"value": 255.}},
        ]
    }
    t['model']['params'] = {
            'stride': 1,
            'encode_nb_filters': [64, 128, 256],
            'encode_filter_sizes': [5, 5, 5],
            'encode_activations': ['relu', 'relu', 'relu'],
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_channel', 'params': {'stride': 4}},
            ],
            'decode_nb_filters': [256, 256],
            'decode_filter_sizes': [5, 5],
            'decode_activations': ['relu', 'relu'],
            'output_filter_size': 5,
            'output_activation': 'sigmoid'
    }
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 1000,
        'nb_iter': 100,
        'binarize':{
            'name': 'none',
            'params': {
                'one_ratio': 0.13,
                'is_moving': True,
            }
        },
        'noise':{
            'name': 'none',
            'params': {
            }
        },
        'stop_if_unchanged': False,
        'seed': 42,
    }
    return t, g


def _set_folder(t, g, folder):
    folder = os.path.join(base_folder, folder)
    t['report']['outdir'] = folder
    g['model']['folder'] = folder
    g['method']['save_folder'] = '{}/gen'.format(folder)
    return t, g
