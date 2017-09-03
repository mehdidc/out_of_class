import os

from clize import run
from collections import OrderedDict

import keras.backend as K

from skimage.io import imsave
import pandas as pd
import numpy as np

from keras.models import Model

from machinedesign.viz import grid_of_images_default
from machinedesign.autoencoder.interface import load

from metrics import compute_max
from metrics import compute_count
from metrics import compute_mmd
from metrics import compute_frechet
from metrics import compute_objectness
from metrics import compute_normalized_entropy

from lightjob.cli import load_db

def evaluate(*, out='jobs.csv'):
    db = load_db('ae/.lightjob')
    dirname = os.path.join('ae', 'results', 'jobs')
    folders = os.listdir(dirname)
    folders = [os.path.join(dirname, folder) for folder in folders]
    df = _evaluate(folders)
    for i in range(len(df)):
        r = df.iloc[i]
        name = r['name']
        stats = r.drop('name').to_dict()
        for k, v in stats.items():
            stats[k] = float(v)
        db.job_update(name, {'stats': stats})


def _evaluate(folders):
    nb = 1000
    theta = 0.9
    digits = np.arange(0, 10)
    letters = np.arange(10, 36)

    data = np.load('data/digits.npz')
    Xtrue_digits = data['X'][0:nb]
    data = np.load('data/letters.npz')
    Xtrue_letters = data['X'][0:nb]

    clf_digits_and_letters = load('discr/digits_and_letters_balanced')
    clf_digits = load('discr/digits')
    clf_letters = load('discr/letters')
    clf_hwrt = load('discr/hwrt')

    enc = Model(
        inputs=clf_digits_and_letters.layers[0].input, 
        outputs=clf_digits_and_letters.get_layer('p_re_lu_4').output
    )
    htrue_digits = enc.predict(Xtrue_digits)
    htrue_letters = enc.predict(Xtrue_letters)
    rows = [] 
    for folder in folders:
        print(folder)
        col = OrderedDict()
        name = os.path.basename(folder)
        col['name'] = name
        model = load(folder)
        col['nb_params'] = model.count_params()
        filename = '{}/gen/generated.npz'.format(folder)
        if not os.path.exists(filename):
            continue
        X = np.load(filename)['generated']
        X = X[0:nb]
        
        probas = clf_digits_and_letters.predict(X)
        col['count_digits'] = compute_count(probas, digits, theta=theta)
        col['count_letters'] = compute_count(probas, letters, theta=theta)
        col['max_digits'] = compute_max(probas, digits, theta=theta)
        col['max_letters'] = compute_max(probas, letters, theta=theta)
        
        col['digits_and_letters_objectness'] = compute_objectness(probas)
        col['digits_and_letters_entropy'] = compute_normalized_entropy(probas)

        probas = clf_digits.predict(X)
        col['digits_objectness'] = compute_objectness(probas)
        col['digits_entropy'] = compute_normalized_entropy(probas)

        probas = clf_letters.predict(X)
        col['letters_objectness'] = compute_objectness(probas)
        col['letters_entropy'] = compute_normalized_entropy(probas)

        probas = clf_hwrt.predict(X)
        col['hwrt_objectness'] = compute_objectness(probas)
        col['hwrt_entropy'] = compute_normalized_entropy(probas)

        h = enc.predict(X)
        col['frechet_digits'] = abs(compute_frechet(h, htrue_digits))
        col['mmd_digits'] = compute_mmd(h, htrue_digits)

        col['frechet_letters'] = abs(compute_frechet(h, htrue_letters))
        col['mmd_letters'] = compute_mmd(h, htrue_letters)
        rows.append(col)
    return pd.DataFrame(rows)


def ppgn(*, generator='mnist', discriminator='digits', layer_name='dense_2', 
         unit_id=0, nb_iter=100, nb_samples=9, eps1=1., eps2=1., out='out.png'):

    ae = load('ae/results/{}'.format(generator))
    discr = load('discr/{}'.format(discriminator))
    x = discr.layers[0].input
    y = discr.get_layer(layer_name).output
    loss = y[:, unit_id].mean()
    grad = K.gradients(loss, x)
    get_grad = K.function([x, K.learning_phase()], grad)
    shape = ae.input_shape[1:]
    X = np.random.uniform(size=(nb_samples,) + shape)
    X = X.astype('float32')
    for _ in range(nb_iter):
        rec = (ae.predict(X) - X)
        X += eps1 * rec + eps2 * get_grad([X, 0])
        print(discr.predict(X)[:, unit_id].mean(), (rec**2).mean())
    im = grid_of_images_default(X)
    imsave(out, im)


if __name__ == '__main__':
    run([evaluate, ppgn])
