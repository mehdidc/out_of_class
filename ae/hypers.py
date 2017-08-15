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
    t["optim"]["max_nb_epochs"] = 3000
    t['model'] = {
        'name': 'fully_connected',
        'params':{
            'nb_hidden_units': [1000],
            'activations': ['relu', {'name': 'winner_take_all_fc', 'params': {'zero_ratio': 0.95}}],
            'output_activation': 'linear',
         }
    }
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 1000,
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


def mnist_dense_sigmoid():
    t, g = mnist_dense()
    t["optim"]["max_nb_epochs"] = 3000
    t['model']['params']['output_activation'] = 'sigmoid'
    t['model']['params']['activations'] = ['relu', {'name': 'winner_take_all_fc', 'params': {'zero_ratio': 0.98}}]
    t['optim']['algo']= {
        'name': 'adam',
        'params': {'lr': 1e-4}
    }
    return t, g


def celeba_dense():
    t, g = mnist_dense()
    t['data']['train'] = {
        'pipeline':[
            {"name": "load_numpy", "params": {"filename": '../data/celeba.npz'}},
            {"name": "divide_by", "params": {"value": 255.}},
        ]
    }
    return t, g



def shoes_dense():
    t, g = mnist_dense()
    t['data']['train'] = {
        'pipeline':[
            {"name": "load_numpy", "params": {"filename": '../data/shoes.npz'}},
            {"name": "divide_by", "params": {"value": 255.}},
        ]
    }
    return t, g

def stl_dense():
    t, g = mnist_dense()
    t['data']['train'] = {
        'pipeline':[
            {"name": "load_numpy", "params": {"filename": '../data/stl.npz'}},
            {"name": "divide_by", "params": {"value": 255.}},
        ]
    }
    return t, g

def celeba_dense():
    t, g=  mnist_dense()
    t['data']['train'] = {
        'pipeline':[
            {"name": "load_numpy", "params": {"filename": '../data/celeba.npz'}},
            {"name": "divide_by", "params": {"value": 255.}},
        ]
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

def celeba64_vertebrate_deep():
    t, g = shoes64_vertebrate_deep()
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/celeba64.npz'
    return t, g


def celeba64_dcgan():
    t, g = shoes64_vertebrate_deep()
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/celeba64.npz'
    t['model'] = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 2,
            'encode_nb_filters': [64, 128, 256, 512],
            'encode_filter_sizes': [5, 5, 5, 5],
            'encode_activations': ['relu', 'relu', 'relu', 'relu'],
            'code_activations': [],
            'decode_nb_filters': [256, 128, 64],
            'decode_filter_sizes': [5, 5, 5],
            'decode_activations': ['relu', 'relu', 'relu'],
            'output_filter_size': 5,
            'output_activation': 'sigmoid'
         }
    }
    return t, g


def celeba64_dcgan_extended():
    t, g = celeba64_dcgan()
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/celeba64.npz'
    t['model'] = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 2,
            'encode_nb_filters': [64, 128, 128, 256, 256, 512],
            'encode_filter_sizes': [5] * 6,
            'encode_activations': ['relu'] * 6,
            'code_activations': [],
            'decode_nb_filters': [256, 256, 128, 128, 64],
            'decode_filter_sizes': [5] * 5,
            'decode_activations': ['relu'] * 5,
            'output_filter_size': 5,
            'output_activation': 'sigmoid'
         }
    }
    return t, g



def celeba64_dcgan_perceptual():
    t, g = celeba64_dcgan()
    t['optim']['loss'] = {
        'name': 'sum',
        'params':{
            'terms':[
                # Pixel space loss
                {
                    'coef': 1.0,
                    'loss': 'mean_squared_error', 
                },
                # VGG16 Feature space loss 
                {
                    'coef': 0.001,
                    'loss': {
                        'name': 'feature_space_mean_squared_error',
                        'params': {
                            'model_filename': '../discr/vgg16/model.h5',
                            'layer': 'block5_pool',
                            #check keras.applications.imagenet_utils.py for more info.
                            'reverse_color_channel': True,
                            'scale': [255., 255., 255.],
                            'bias': [-103.939, -116.779, -123.68], 
                        }
                    }, 
                }
            ]
        }
    }
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

