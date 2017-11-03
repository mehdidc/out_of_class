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
from metrics import compute_normalized_diversity

from lightjob.cli import load_db

def evaluate(*, force=False):
    db = load_db('ae/.lightjob')
    folders = []
    for j in db.jobs_with():
        if ('stats' not in j or j['stats'] is None) or force:
            dirname = os.path.join('ae', 'results', 'jobs', j['summary'])
            folders.append(dirname)
    df = _evaluate(folders)
    for i in range(len(df)):
        r = df.iloc[i]
        name = r['name']
        stats = r.drop('name').to_dict()
        for k, v in stats.items():
            stats[k] = float(v)
        print(stats)
        db.job_update(name, {'stats': stats})


def _evaluate(folders):
    nb = 1000
    theta = 0.9
    digits = np.arange(0, 10)
    letters = np.arange(10, 36)
    emnist_digits = np.arange(0, 10)
    emnist_letters = np.arange(10, 47)

    data = np.load('data/digits.npz')
    Xtrue_digits = data['X'][0:nb]
    data = np.load('data/letters.npz')
    Xtrue_letters = data['X'][0:nb]

    clf_digits_and_letters = load('discr/digits_and_letters_balanced')
    clf_digits = load('discr/digits')
    clf_letters = load('discr/letters')
    clf_hwrt = load('discr/hwrt')
    clf_emnist = load('discr/emnist')

    enc = Model(
        inputs=clf_digits_and_letters.layers[0].input, 
        outputs=clf_digits_and_letters.get_layer('p_re_lu_4').output
    )
    htrue_digits = enc.predict(Xtrue_digits)
    htrue_letters = enc.predict(Xtrue_letters)
    rows = [] 
    for folder in folders:
        if not os.path.exists(os.path.join(folder, 'model.h5')):
            continue
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
        col['digits_count'] = compute_count(probas, digits, theta=theta)
        col['letters_count'] = compute_count(probas, letters, theta=theta)
        col['digits_max'] = compute_max(probas, digits, theta=theta)
        col['letters_max'] = compute_max(probas, letters, theta=theta)
       
        col['digits_and_letters_objectness'] = compute_objectness(probas)
        col['digits_and_letters_entropy'] = compute_normalized_entropy(probas)

        probas = clf_emnist.predict(X)
        col['emnist_digits_count'] = compute_count(probas, emnist_digits, theta=theta)
        col['emnist_letters_count'] = compute_count(probas, emnist_letters, theta=theta)
        col['emnist_digits_max'] = compute_max(probas, emnist_digits, theta=theta)
        col['emnist_letters_max'] = compute_max(probas, emnist_letters, theta=theta)

        col['emnist_objectness'] = compute_objectness(probas)
        col['emnist_letters_entropy'] = compute_normalized_entropy(probas)

        probas = clf_digits.predict(X)
        col['digits_objectness'] = compute_objectness(probas)
        col['digits_entropy'] = compute_normalized_entropy(probas)
        col['digits_diversity'] = compute_normalized_diversity(probas)

        probas = clf_letters.predict(X)
        col['letters_objectness'] = compute_objectness(probas)
        col['letters_entropy'] = compute_normalized_entropy(probas)
        col['letters_diversity'] = compute_normalized_diversity(probas)

        probas = clf_hwrt.predict(X)
        col['hwrt_objectness'] = compute_objectness(probas)
        col['hwrt_entropy'] = compute_normalized_entropy(probas)
        col['hwrt_diversity'] = compute_normalized_diversity(probas)

        h = enc.predict(X)
        col['digits_frechet'] = abs(compute_frechet(h, htrue_digits))
        col['digits_mmd'] = compute_mmd(h, htrue_digits)

        col['letters_frechet'] = abs(compute_frechet(h, htrue_letters))
        col['letters_mmd'] = compute_mmd(h, htrue_letters)
        rows.append(col)
    return pd.DataFrame(rows)


def ppgn(*, generator='mnist', discriminator='digits_and_letters', layer_name='dense_2', 
         unit_id=0, nb_iter=100, nb_samples=9, eps1=1., eps2=1., eps3=0., out='out.png'):

    ae = load('ae/results/{}'.format(generator))
    discr = load('discr/{}'.format(discriminator))
    discr.summary()
    x = discr.layers[0].input
    y = discr(x)
    loss = y[:, unit_id].mean()
    grad = K.gradients(loss, x)
    get_grad = K.function([x, K.learning_phase()], grad)
    shape = ae.input_shape[1:]
    X = np.random.uniform(size=(nb_samples,) + shape)
    X = X.astype('float32')
    for _ in range(nb_iter):
        rec = (ae.predict(X) - X)
        X += (eps1 / (len(X))) * rec + eps2 * get_grad([X, 0]) + eps3 * np.random.normal(0, 1, size=X.shape)
        np.clip(X, 0, 1)
        print('confidence : {:.2f}, reconstruction : {:.4f}'.format(discr.predict(X)[:, unit_id].mean(), (rec**2).mean()))
    im = grid_of_images_default(X)
    imsave(out, im)


def extract(*, generator='mnist', discriminator='letters', classes=None, out='extracted.png', nb=10):
    data = np.load('ae/results/{}/gen/generated.npz'.format(generator))
    X = data['generated']
    discr = load('discr/{}'.format(discriminator))
    y = discr.predict(X)

    if classes is None:
        classes = np.arange(y.shape[1])
    else:
        first, last = classes.split('-')
        first = int(first)
        last = int(last)
        classes = np.arange(first, last)

    x = []
    for c in classes:
        idx = y[:, c].argsort()[::-1]
        x.append(X[idx][0:nb])
    x = np.array(x)
    x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
    im = grid_of_images_default(x, shape=(len(classes), nb))
    imsave(out, im)


if __name__ == '__main__':
    run([evaluate, ppgn, extract])
