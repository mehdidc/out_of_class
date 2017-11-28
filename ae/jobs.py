from hypers import mnist

def mnist_basic(rng):
    nb = int(rng.choice((1, 2, 3, 4, 5, 6)))
    k = 5
    fsize = (k - 1) * nb + 1
    stride = 1
    t, g = mnist()
    t['report']['outdir'] = ''
    g['model']['folder'] = ''
    g['method']['save_folder'] =''
    t['model'] = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [128] * nb,
            'encode_filter_sizes': [k] * nb,
            'encode_activations': ['relu'] * nb,
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_channel', 'params': {'stride': stride}},
            ],
            'decode_nb_filters': [],
            'decode_filter_sizes': [],
            'decode_activations': [],
            'output_filter_size': fsize,
            'output_activation': 'sigmoid'
         }
    }
    g['method']['params'] = {
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
    }
    return t, g



def mnist_deep(rng):
    nb = int(rng.choice((1, 2, 3, 4, 5, 6)))
    stride = int(rng.choice((0, 1, 2, 4)))
    t, g = mnist()
    t['report']['outdir'] = ''
    g['model']['folder'] = ''
    g['method']['save_folder'] =''
    t['model'] = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [128] * nb,
            'encode_filter_sizes': [5] * nb,
            'encode_activations': ['relu'] * nb,
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_channel', 'params': {'stride': stride}},
            ],
            'decode_nb_filters': [128] * (nb - 1),
            'decode_filter_sizes': [5] * (nb - 1),
            'decode_activations': ['relu'] * (nb - 1),
            'output_filter_size': 5,
            'output_activation': 'sigmoid'
         }
    }
    g['method']['params'] = {
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
    }
    return t, g


def mnist_deep_lifetime(rng):
    nb = int(rng.choice((1, 2, 3, 4, 5, 6)))
    zero_ratio = rng.choice((0.2, 0.4, 0.7, 0.9))
    t, g = mnist()
    t['report']['outdir'] = ''
    g['model']['folder'] = ''
    g['method']['save_folder'] =''
    t['model'] = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [128] * nb,
            'encode_filter_sizes': [5] * nb,
            'encode_activations': ['relu'] * nb,
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_lifetime', 'params': {'zero_ratio': zero_ratio}},
            ],
            'decode_nb_filters': [128] * (nb - 1),
            'decode_filter_sizes': [5] * (nb - 1),
            'decode_activations': ['relu'] * (nb - 1),
            'output_filter_size': 5,
            'output_activation': 'sigmoid'
         }
    }
    g['method']['params'] = {
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
    }
    return t, g


def mnist_deep_kchannel(rng):
    nb = int(rng.choice((1, 2, 3, 4, 5, 6)))
    zero_ratio = rng.choice((0.1, 0.2, 0.4, 0.5, 0.7, 0.9))
    #zero_ratio = rng.choice((0.9, 0.7))
    t, g = mnist()
    t['report']['outdir'] = ''
    g['model']['folder'] = ''
    g['method']['save_folder'] =''
    t['model'] = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [128] * nb,
            'encode_filter_sizes': [5] * nb,
            'encode_activations': ['relu'] * nb,
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_kchannel', 'params': {'zero_ratio': zero_ratio}},
            ],
            'decode_nb_filters': [128] * (nb - 1),
            'decode_filter_sizes': [5] * (nb - 1),
            'decode_activations': ['relu'] * (nb - 1),
            'output_filter_size': 5,
            'output_activation': 'sigmoid'
         }
    }
    g['method']['params'] = {
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
    }
    return t, g

def mnist_capacity(rng):
    nb = int(rng.choice((1, 2, 3, 4, 5, 6)))
    bottleneck = int(rng.choice((64, 32, 16, 8, 4, 2)))
    t, g = mnist()
    t['report']['outdir'] = ''
    g['model']['folder'] = ''
    g['method']['save_folder'] = ''
    t['model'] = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [128] * (nb - 1) + [bottleneck],
            'encode_filter_sizes': [5] * nb,
            'encode_activations': ['relu'] * nb,
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
            ],
            'decode_nb_filters': [128] * (nb - 1),
            'decode_filter_sizes': [5] * (nb - 1),
            'decode_activations': ['relu'] * (nb - 1),
            'output_filter_size': 5,
            'output_activation': 'sigmoid'
         }
    }
    g['method']['params'] = {
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
    }
    return t, g


def mnist_noise(rng):
    stride = 4
    nb = int(rng.choice((1, 2, 3, 4, 5, 6)))
    proba = rng.choice((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    proba = float(proba)
    t, g = mnist()
    t['report']['outdir'] = ''
    g['model']['folder'] = ''
    g['method']['save_folder'] =''
    model = {
        'name': 'convolutional_bottleneck',
        'params':{
            'stride': 1,
            'encode_nb_filters': [128] * nb,
            'encode_filter_sizes': [5] * nb,
            'encode_activations': ['relu'] * nb,
            'code_activations': [
                {'name': 'winner_take_all_spatial', 'params': {}},
                {'name': 'winner_take_all_channel', 'params': {'stride': stride}},
            ],
            'decode_nb_filters': [128] * (nb - 1),
            'decode_filter_sizes': [5] * (nb - 1),
            'decode_activations': ['relu'] * (nb - 1),
            'output_filter_size': 5,
            'output_activation': 'sigmoid'
         }
    }
    t['model'] = [
        {'name': 'noise', 'params':{'type': 'salt_and_pepper', 'params':{'proba': proba}}},
        model
    ]
    g['method']['params'] = {
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
    }
    return t, g
