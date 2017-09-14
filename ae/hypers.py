basic_train_params = {
    'family': 'autoencoder',
    'input_col': 'X',
    'output_col': 'X',
    'model': {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [128, 128, 128],
            'encode_filter_sizes': [5, 5, 5],
            'encode_activations': ['relu', 'relu', 'relu'],
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_channel', 'params': {'stride': 4}},
            ],
            'decode_nb_filters': [128, 128],
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
        'budget_secs': 3600 * 24 * 5,
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
            'nb_samples': 1000,
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


def mnist_basic():
    # ICCC model
    t, g = mnist()
    t['model'] = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [128, 128, 128],
            'encode_filter_sizes': [5, 5, 5],
            'encode_activations': ['relu', 'relu', 'relu'],
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_channel', 'params': {'stride': 4}},
            ],
            'decode_nb_filters': [],
            'decode_filter_sizes': [],
            'decode_activations': [],
            'output_filter_size': 13,
            'output_activation': 'sigmoid'
         }
    }
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


def iccc():
    t, g = mnist_basic()
    t['model']['params']['output_activation'] = 'sigmoid'
    return t, g 


def mnist():
    t = basic_train_params.copy()
    g = basic_generate_params.copy()
    return t, g


def mnist_without_sparsity():
    t, g = mnist()
    t['model']['params'] = {
        'stride': 1,
        'encode_nb_filters': [16, 16, 16],
        'encode_filter_sizes': [5, 5, 5],
        'encode_activations': ['relu', 'relu', 'relu'],
        'code_activations': [
        ],
        'decode_nb_filters': [16, 16],
        'decode_filter_sizes': [5, 5],
        'decode_activations': ['relu', 'relu'],
        'output_filter_size': 5,
        'output_activation': 'sigmoid'
    }
    return t, g

def mnist2():
    return mnist()

def mnist3():
    nb = 6
    t, g = mnist()
    t['model'] = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [128] * (nb - 1) + [256],
            'encode_filter_sizes': [5] * nb,
            'encode_activations': ['relu'] * nb,
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_channel', 'params': {'stride': 1}},
            ],
            'decode_nb_filters': [256] + [128] * (nb - 2),
            'decode_filter_sizes': [5] * (nb - 1),
            'decode_activations': ['relu'] * (nb - 1),
            'output_filter_size': 5,
            'output_activation': 'sigmoid'
         }
    }
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

def mnist4():
    nb = 6
    t, g = mnist()
    t['model'] = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [128] * (nb - 1) + [256],
            'encode_filter_sizes': [5] * nb,
            'encode_activations': ['relu'] * nb,
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_kchannel', 'params': {'zero_ratio': 0.8}},
            ],
            'decode_nb_filters': [256] + [128] * (nb - 2),
            'decode_filter_sizes': [5] * (nb - 1),
            'decode_activations': ['relu'] * (nb - 1),
            'output_filter_size': 5,
            'output_activation': 'sigmoid'
         }
    }
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



def mnist_with_denoising():
    t, g = mnist()
    t['model'] = [
        {'name': 'noise', 'params':{'type': 'salt_and_pepper', 'params':{'proba': 0.5}}},
        t['model'],
    ]
    return t, g


def mnist_with_denoising2():
    t, g = mnist()
    t['model'] = [
        {'name': 'noise', 'params':{'type': 'salt_and_pepper', 'params':{'proba': 0.3}}},
        t['model'],
    ]
    return t, g



def mnist_dense():
    # sparse with winner take all fc (mini-batch based)
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


def mnist_dense2():
    t, g = mnist()
    t["optim"]["max_nb_epochs"] = 3000
    t['model'] = [
        {'name': 'noise', 'params':{'type': 'salt_and_pepper', 'params':{'proba': 0.5}}},
        {
            'name': 'fully_connected',
            'params':{
                'nb_hidden_units': [1000],
                'activations': ['relu'],
                'output_activation': 'sigmoid',
             }
        }
    ]
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 10000,
        'nb_iter': 100,
        'noise':{
            'name': 'salt_and_pepper',
            'params': {
                'proba': 0.3
            }
        },
        'binarize':{
            'name': 'none',
            'params': {
            }
        },
        'stop_if_unchanged': False,
        'seed': 42,
    }
    return t, g

def mnist_dense3():
    t, g = mnist()
    t["optim"]["max_nb_epochs"] = 3000
    t['model'] = {
        'name': 'fully_connected',
        'params':{
            'nb_hidden_units': [1000, 2000],
            'activations': ['relu', {'name': 'winner_take_all_fc', 'params': {'zero_ratio': 0.85}}],
            'output_activation': 'sigmoid',
         }
    }
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 100,
        'nb_iter': 100,
        'binarize':{
            'name': 'binary_threshold',
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



def mnist_dcgan():
    # reduce size of space to 1 in the bottleneck
    t = basic_train_params.copy()
    g = basic_generate_params.copy()
    t['model'] = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [16, 32, 64],
            'encode_filter_sizes': [10, 10, 10],
            'encode_activations': ['relu', 'relu', 'relu'],
            'code_activations': [],
            'decode_nb_filters': [32, 16],
            'decode_filter_sizes': [10, 10],
            'decode_activations': ['relu', 'relu'],
            'output_filter_size': 10,
            'output_activation': 'sigmoid'
         }
    }
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 10000,
        'nb_iter': 100,
        'noise':{
            'name': 'salt_and_pepper',
            'params': {
                'proba': 0.3
            }
        },
        'binarize':{
            'name': 'none',
            'params': {
            }
        },
        'stop_if_unchanged': False,
        'seed': 42,
    }
    return t, g


def flaticon_data(nc=32):
    c = '' if nc == 32 else '64'
    return {
        'train': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": '../data/flaticon{}.npz'.format(c)}},
                {"name": "divide_by", "params": {"value": 255.}},
            ]
        },
        'transformers':[
        ]
    }


def flaticon():
    t, g = mnist()
    t['data'] = flaticon_data()
    t['model']  = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [128, 256, 512],
            'encode_filter_sizes': [5] * 3,
            'encode_activations': ['relu'] * 3,
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_channel', 'params': {'stride': 4}},
            ],
            'decode_nb_filters': [],
            'decode_filter_sizes': [],
            'decode_activations': [],
            'output_filter_size': 13,
            'output_activation': 'sigmoid'
         }
    }
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 100,
        'nb_iter': 100,
        'binarize':{
            'name': 'binary_threshold',
            'params': {
                'one_ratio': 0.2,
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

    return t,g


def flaticon_deep():
    t, g = mnist()
    t['data'] = flaticon_data()
    t['model']  = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [128, 256, 512],
            'encode_filter_sizes': [5] * 3,
            'encode_activations': ['relu'] * 3,
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_channel', 'params': {'stride': 1}},
            ],
            'decode_nb_filters': [256, 128],
            'decode_filter_sizes': [5, 5],
            'decode_activations': ['relu', 'relu'],
            'output_filter_size': 5,
            'output_activation': 'sigmoid'
         }
    }
    return t,g




def flaticon_vertebrate():
    t, g = mnist_vertebrate()
    t['data'] = flaticon_data()
    return t, g


def flaticon64_vertebrate():
    t, g = shoes64_vertebrate() 
    t['data'] = flaticon_data(nc=64)
    return t, g


def flaticon64_vertebrate_deep():
    t, g = shoes64_vertebrate_deep() 
    t['data'] = flaticon_data(nc=64)
    t['optim']['batch_size'] = 32
    t['optim']['pred_batch_size'] = 32
    return t, g


def flaticon64_vertebrate_deep_without_sparsity():
    t, g = shoes64_vertebrate_deep() 
    t['data'] = flaticon_data(nc=64)
    t['optim']['batch_size'] = 32
    t['optim']['pred_batch_size'] = 32
    t['model']['params']['code_activations'] = []
    return t, g


def flaticon64_vertebrate_deep2():
    t, g = shoes64_vertebrate_deep() 
    t['data'] = flaticon_data(nc=64)
    t['optim']['batch_size'] = 32
    t['optim']['pred_batch_size'] = 32

    t['model'] = {
        'name': 'vertebrate',
        'params': {
            'encode_stride': 2,
            'encode_nb_filters': [256, 256, 256, 256],
            'encode_filter_sizes': [5, 5, 5, 5],
            'encode_activations': ['relu', 'relu', 'relu', 'relu'],
            'code_activations': [
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
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
                {
                    'nb_filters': [256, 256, 256],
                    'filter_sizes': [5, 5, 5],
                    'activations': ['relu', 'relu', 'relu'],
                    'output_filter_size': 5,
                    'stride': 2,
                },
            ],
            'output_activation': 'sigmoid',
        },
    } 
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 100,
        'nb_iter': 100,
        'binarize':{
            'name': 'binary_threshold',
            'params': {
                'one_ratio': 0.3,
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


def flaticon64_vertebrate_deep3():
    t, g = shoes64_vertebrate_deep() 
    t['data'] = flaticon_data(nc=64)
    t['optim']['batch_size'] = 32
    t['optim']['pred_batch_size'] = 32

    t['model'] = {
        'name': 'vertebrate',
        'params': {
            'encode_stride': 2,
            'encode_nb_filters': [256, 256, 256, 256],
            'encode_filter_sizes': [5, 5, 5, 5],
            'encode_activations': ['relu', 'relu', 'relu', 'relu'],
            'code_activations': [
                    [{'name': 'winner_take_all_spatial', 'params': {}}],
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
                {
                    'nb_filters': [],
                    'filter_sizes': [],
                    'activations': [],
                    'output_filter_size': 17,
                    'stride': 16,
                },
            ],
            'output_activation': 'sigmoid',
        },
    }
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 100,
        'nb_iter': 100,
        'binarize':{
            'name': 'binary_threshold',
            'params': {
                'one_ratio': 0.2,
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


def mnist_dcgan_nontrained():
    t, g = mnist_dcgan()
    t['optim']['algo']= {
        'name': 'adam',
        'params': {'lr': 0.0}
    }
    return t, g


def mnist_nontrained():
    t, g = mnist()
    t['optim']['algo']= {
        'name': 'adam',
        'params': {'lr': 0.0}
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


def mnist_vertebrate2():
    t, g = mnist()
    t['model'] = {
        'name': 'vertebrate',
        'params': {
            'encode_stride': 1,
            'encode_nb_filters': [64, 128, 256],
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
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 100,
        'nb_iter': 100,
        'binarize':{
            'name': 'binary_threshold',
            'params': {
                'one_ratio': 0.2,
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
    g['method']['params'] = {
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


def celeba64aligned_dcgan():
    t, g = celeba64_dcgan()
    t['data']['train']['pipeline'] = [
        {"name": "load_hdf5", "params": {"filename": "../data/celeba64_align.h5", "nb": 30000}},
        {"name": "divide_by", "params": {"value": 255.}},
    ]
    return t, g

def celeba64aligned_vertebrate_deep():
    t, g = celeba64_vertebrate_deep()
    t['data']['train']['pipeline'] = [
        {"name": "load_hdf5", "params": {"filename": "../data/celeba64_align.h5", "nb": 30000}},
        {"name": "divide_by", "params": {"value": 255.}},
    ]
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

def hwrt_padded():
    t, g = mnist()
    dataset = '../data/hwrt_padded.npz'
    t['data']['train'] = {
        'pipeline':[
            {"name": "load_numpy", "params": {"filename": dataset}},
            {"name": "divide_by", "params": {"value": 255.}},
        ]
    }
    return t, g 
 

def hwrt_dcgan():
    t, g = mnist_dcgan()
    dataset = '../data/hwrt.npz'
    t['data']['train'] = {
        'pipeline':[
            {"name": "load_numpy", "params": {"filename": dataset}},
            {"name": "divide_by", "params": {"value": 255.}},
        ]
    }
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 100,
        'nb_iter': 100,
        'binarize':{
            'name': 'binary_threshold',
            'params': {
                'value': .6,
                'is_moving': False,
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
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/celeba.npz'
    return t, g

def celeba2_discrete():
    t, g = celeba_discrete()
    t['model']['params'] = {
        'stride': 1,
        'encode_nb_filters': [64, 128, 256],
        'encode_filter_sizes': [5, 5, 5],
        'encode_activations': ['relu', 'relu', 'relu'],
        'code_activations': [
            {'name': 'winner_take_all_spatial', 'params': {}},
            {'name': 'winner_take_all_channel', 'params': {'stride': 2}},
        ],
        'decode_nb_filters': [256, 256],
        'decode_filter_sizes': [5, 5],
        'decode_activations': ['relu', 'relu'],
        'output_filter_size': 5,
        'output_activation': {'name': 'axis_softmax', 'params': {'axis': 1}},
    }
    return t, g

def shoes_discrete():
    t, g = cifar_discrete()
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/shoes.npz'
    return t, g

def shoes_discrete2():
    t, g = cifar_discrete()
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/shoes.npz'
    t['model']['params'] = {
        'stride': 2,
        'encode_nb_filters': [64, 64, 128, 128, 256],
        'encode_filter_sizes': [5] * 5,
        'encode_activations': ['relu'] * 5,
        'code_activations': [
        ],
        'decode_nb_filters': [128, 128, 128, 256] ,
        'decode_filter_sizes': [5] * 4,
        'decode_activations': ['relu'] * 4,
        'output_filter_size': 5,
        'output_activation': {'name': 'axis_softmax', 'params': {'axis': 1}},
    }
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 100,
        'nb_iter': 100,
        'binarize':{
            'name': 'onehot',
            'params': {
                'axis': 1
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


def mnist_discrete():
    t, g = cifar_discrete()
    t['data']['train']['pipeline'][0]['params']['filename'] = '../data/digits.npz'
    t['data']['transformers'] = [
        {'name': 'ColorDiscretizer', 'params': {'nb_centers': 3}}
    ]
    t['model']['params']['code_activations'] = [
        {'name': 'winner_take_all_spatial', 'params': {}},
        {'name': 'winner_take_all_channel', 'params': {'stride': 4}},
    ]
    g['method']['params'] = {
        'batch_size': 128,
        'nb_samples': 100,
        'nb_iter': 100,
        'binarize':{
            'name': 'onehot',
            'params': {
                'axis': 1
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
