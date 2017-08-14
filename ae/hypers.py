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


def mnist():
    t = basic_train_params.copy()
    g = basic_generate_params.copy()
    return t, g

def mnist_without_sparsity():
    t, g = mnist()
    t['model']['params']['code_activations']  = []
    return t, g

def mnist_dense():
    t, g = mnist()
    t["optim"]["max_nb_epochs"] = 1000
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
                'activations': ['relu'] * 5 + [{'name': 'ksparse', 'params': {'zero_ratio': 0.74}}],
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


def mnist_vertebrate():
    t, g = mnist()
    t['model'] = {
        'name': 'vertebrate',
        'params': {
            'encode_stride': 1,
            'encode_nb_filters': [64, 64, 64],
            'encode_filter_sizes': [5, 5, 5],
            'encode_activations': ['relu', 'relu', 'relu'],
            'code_activations': [
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
            ],
            'decode': [
                {
                    'nb_filters': [],
                    'filter_sizes': [],
                    'activations': [],
                    'output_filter_size': 5,
                    'stride': 1,
                },
                {
                    'nb_filters': [],
                    'filter_sizes': [],
                    'activations': [],
                    'output_filter_size': 9,
                    'stride': 1,
                },
                {
                    'nb_filters': [],
                    'filter_sizes': [],
                    'activations': [],
                    'output_filter_size': 13,
                    'stride': 1,
                },
            ],
            'output_activation': 'sigmoid',
        },
    }
    return t, g


def mnist_vertebrate_deep():
    t, g = mnist()
    t['model'] = {
        'name': 'vertebrate',
        'params': {
            'encode_stride': 1,
            'encode_nb_filters': [64, 64, 64],
            'encode_filter_sizes': [5, 5, 5],
            'encode_activations': ['relu', 'relu', 'relu'],
            'code_activations': [
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
            ],
            'decode': [
                {
                    'nb_filters': [],
                    'filter_sizes': [],
                    'activations': [],
                    'output_filter_size': 5,
                    'stride': 1,
                },
                {
                    'nb_filters': [64],
                    'filter_sizes': [5],
                    'activations': ['relu'],
                    'output_filter_size': 5,
                    'stride': 1,
                },
                {
                    'nb_filters': [64, 64],
                    'filter_sizes': [5, 5],
                    'activations': ['relu', 'relu'],
                    'output_filter_size': 5,
                    'stride': 1,
                },
            ],
            'output_activation': 'sigmoid',
        },
    }
    return t, g


def mnist_vertebrate_binary():
    t, g = mnist_vertebrate()
    t['model']['params']['code_activations'] = [
        [{'name': 'winner_take_all_spatial', 'params': {}}, {'name': 'MinMaxNormalizer', 'params':{'axis': 0}}],
        [{'name': 'winner_take_all_spatial', 'params': {}}, {'name': 'MinMaxNormalizer', 'params':{'axis': 0}}],
        [{'name': 'winner_take_all_spatial', 'params': {}}, {'name': 'MinMaxNormalizer', 'params':{'axis': 0}}],
    ]
    return t, g 

def mnist_vertebrate_deep_binary():
    t, g = mnist_vertebrate_deep()
    t['model']['params']['code_activations'] = [
        [{'name': 'winner_take_all_spatial', 'params': {}}, {'name': 'MinMaxNormalizer', 'params':{'axis': 0}}],
        [{'name': 'winner_take_all_spatial', 'params': {}}, {'name': 'MinMaxNormalizer', 'params':{'axis': 0}}],
        [{'name': 'winner_take_all_spatial', 'params': {}}, {'name': 'MinMaxNormalizer', 'params':{'axis': 0}}],
    ]
    return t, g 

def shoes_vertebrate():
    t, g = mnist_vertebrate()
    t['data']['train'] = {
        'pipeline':[
            {"name": "load_numpy", "params": {"filename": '../data/shoes.npz'}},
            {"name": "divide_by", "params": {"value": 255.}},
        ]
    }
    t['model'] = {
        'name': 'vertebrate',
        'params': {
            'encode_stride': 2,
            'encode_nb_filters': [256, 256, 256],
            'encode_filter_sizes': [5, 5, 5],
            'encode_activations': ['relu', 'relu', 'relu'],
            'code_activations': [
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
            ],
            'decode': [
                {
                    'nb_filters': [],
                    'filter_sizes': [],
                    'activations': [],
                    'output_filter_size': 5,
                    'stride': 2,
                },
                {
                    'nb_filters': [],
                    'filter_sizes': [],
                    'activations': [],
                    'output_filter_size': 9,
                    'stride': 4,
                },
                {
                    'nb_filters': [],
                    'filter_sizes': [],
                    'activations': [],
                    'output_filter_size': 13,
                    'stride': 8,
                },
            ],
            'output_activation': 'sigmoid',
        },
    }
    return t, g


def shoes_vertebrate_deep():
    t, g = mnist_vertebrate()
    t['data']['train'] = {
        'pipeline':[
            {"name": "load_numpy", "params": {"filename": '../data/shoes.npz'}},
            {"name": "divide_by", "params": {"value": 255.}},
        ]
    }
    t['model'] = {
        'name': 'vertebrate',
        'params': {
            'encode_stride': 2,
            'encode_nb_filters': [256, 256, 256],
            'encode_filter_sizes': [5, 5, 5],
            'encode_activations': ['relu', 'relu', 'relu'],
            'code_activations': [
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
            ],
            'decode': [
                {
                    'nb_filters': [],
                    'filter_sizes': [],
                    'activations': [],
                    'output_filter_size': 5,
                    'stride': 2,
                },
                {
                    'nb_filters': [256],
                    'filter_sizes': [5],
                    'activations': ['relu'],
                    'output_filter_size': 5,
                    'stride': 2,
                },
                {
                    'nb_filters': [256, 256],
                    'filter_sizes': [5, 5],
                    'activations': ['relu', 'relu'],
                    'output_filter_size': 5,
                    'stride': 2,
                },
            ],
            'output_activation': 'sigmoid',
        },
    }
    return t, g


def shoes64_vertebrate():
    t, g = shoes_vertebrate()
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/shoes64.npz' 
    return t, g

def shoes64_vertebrate_deep():
    t, g = shoes_vertebrate_deep()
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/shoes64.npz' 
    return t, g



def celeba_vertebrate():
    t, g = shoes_vertebrate()
    ['data']['train']['pipeline'][0]['params']['filename'] = '../data/celeba.npz' 
    return t, g


def celeba_vertebrate_deep():
    t, g = shoes_vertebrate_deep()
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/celeba.npz'
    return t, g


def hwrt():
    t, g = mnist()
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


def cifar_discrete():
    t, g = mnist()
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
                {"name": "load_numpy", "params": {"filename": '../data/cifar10.npz'}},
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


def svhn_discrete():
    t, g = cifar_discrete()
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/svhn.npz'
    return t, g

def celeba_discrete():
    t, g = cifar_discrete()
    dataset = '../data/celeba.npz'
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/celeba.npz'
    return t, g

def shoes_discrete():
    t, g = cifar_discrete()
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/shoes.npz'
    return t, g

