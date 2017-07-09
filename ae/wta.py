from clize import run
import numpy as np
from skimage.io import imsave

from machinedesign.viz import grid_of_images_default

from machinedesign.autoencoder.interface import train as train_ae
from machinedesign.autoencoder.interface import generate as generate_ae

dataset = '../data/digits.npz'

def train():
    params = {
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
                'decode_nb_filters': [],
                'decode_filter_sizes': [],
                'decode_activations': [],

                'output_filter_size': 13,
                'output_activation': 'sigmoid'
             }
        },
        'data': {
            'train': {
                'pipeline':[
                    {"name": "load_numpy", "params": {"filename": dataset, "nb": 1000}},
                    {"name": "divide_by", "params": {"value": 255.}},
                ]
            },
            'transformers':[
            ]
        },
        'report':{
            'outdir': 'wta',
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
            'batch_size': 128,
            'pred_batch_size': 128,
            "loss": "mean_squared_error",
            'budget_secs': 86400,
            'seed': 42
        },
    }
    train_ae(params)


def generate():
    params = {
        'model':{
            'folder': 'wta'
        },
        'method':{
            'name': 'iterative_refinement',
            'params': {
                'batch_size': 128,
                'nb_samples': 128,
                'nb_iter': 100,
                'binarize':{
                    'name': 'none',
                    'params': {
                        "axis": 1
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
            'save_folder': 'wta/gen',
        }
    }
    generate_ae(params)
    data = np.load('wta/gen/generated.npz')
    X = data['full']
    X = X.transpose((1, 0, 2, 3, 4))
    nb_iter, nb_examples = X.shape[0], X.shape[1]
    X = X.reshape((nb_iter * nb_examples,) + X.shape[2:])
    img = grid_of_images_default(X, shape=(nb_iter, nb_examples))
    imsave('wta/samples.png', img)


if __name__ == '__main__':
    train()
    generate()
