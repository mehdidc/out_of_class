from clize import run
import os

import numpy as np
from skimage.io import imsave

from machinedesign.viz import grid_of_images_default

from interface import train as _train
from interface import generate as _generate

import hypers

base_folder = 'results'

def train(job):
    get_params = getattr(hypers, job)
    t, g = get_params()
    t, g = _set_folder(t, g, job)
    if os.path.exists(t['report']['outdir']):
        print('Repository "{}" exists, I dont override for safety. Delete it if you want to.'.format(t['report']['outdir']))
        return
    _train(t)

def generate(job):
    get_params = getattr(hypers, job)
    t, g = get_params()
    t, g = _set_folder(t, g, job)
    _generate(g)
    data = np.load('{}/generated.npz'.format(g['method']['save_folder']))
    X = data['full']
    X = X.transpose((1, 0, 2, 3, 4))
    nb_iter, nb_examples = X.shape[0], X.shape[1]
    X = X.reshape((nb_iter * nb_examples,) + X.shape[2:])
    img = grid_of_images_default(X, shape=(nb_iter, nb_examples))
    imsave('{}/samples.png'.format(g['model']['folder']), img)

def _set_folder(t, g, folder):
    folder = os.path.join(base_folder, folder)
    t['report']['outdir'] = folder
    g['model']['folder'] = folder
    g['method']['save_folder'] = '{}/gen'.format(folder)

    return t, g
if __name__ == '__main__':
    run([train, generate])
