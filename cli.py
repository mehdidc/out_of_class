import os

from clize import run
from functools import lru_cache

import keras.backend as K

from skimage.io import imsave
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

load = lru_cache(maxsize=None)(load)
np.load = lru_cache(maxsize=None)(np.load)

def _recons_ratio(folder, **kw):
    stat = kw['stats'].get('recons_ratio', {})
    force = kw['force']
    datasets = ['hwrt_thin', 'digits', 'digits_test']
    if all([d in stat for d in datasets]) and not force:
        print('skip')
        return {}
    theta = 50 
    nb = 10000
    out = {}
    model = load(folder)
    for ds in datasets:
        data = np.load('data/{}.npz'.format(ds))
        X = data['X'][0:nb] / 255.0
        Xr = model.predict(X)
        err = (np.abs(X-Xr).sum(axis=(1, 2, 3)) < theta).mean()
        err = float(err)
        out[ds] = err
    print(out)
    return {'recons': out}


def _ratio_unique(folder, **kw):
    stat = kw['stats'].get('recons_ratio', {})
    force = kw['force']
    if 'ratio_unique' in stat and not force and stat['ratio_unique'] is not None:
        print('skip')
        return {}
 
    filename = '{}/gen/generated.npz'.format(folder)
    if not os.path.exists(filename):
        return {}
    X = np.load(filename)['generated']
    X = X > 0.5
    X = X.astype('int32')
    X = [tuple(x.flatten().tolist()) for x in X]
    ratio = len(set(X)) / len(X)
    ratio = float(ratio)
    return {'ratio_unique': ratio}


def _metrics(folder, **kw):
    stat = kw['stats'].get('metrics', {})
    force = kw['force']
    if len(stat) and not force:
        print('skip')
        return {}
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
    if not os.path.exists(os.path.join(folder, 'model.h5')):
        return {}
    col = {}
    model = load(folder)
    col['nb_params'] = model.count_params()
    filename = '{}/gen/generated.npz'.format(folder)
    if not os.path.exists(filename):
        return {}
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
    for k, v in col.items():
        col[k] = float(v)
    return {'metrics': col}

eval_funcs = {
    'ratio_unique': _ratio_unique,
    'metrics': _metrics,
    'recons_ratio': _recons_ratio,
}


def evaluate(*, force=False, name=None, job=None):
    db = load_db('ae/.lightjob')
    if job is not None:
        jobs = db.jobs_with(summary=job)
    else:
        jobs = db.all_jobs()
    for j in jobs:
        folder = os.path.join('ae', 'results', 'jobs', j['summary'])
        stats = {}
        if j.get('stats') is not None:
            stats.update(j['stats'])
        if name is None:
            funcs = eval_funcs.items()
        else:
            funcs = (name, eval_funcs[name]),
        stats_orig = stats.copy() 
        for name, func in funcs:
            print('Eval of {:<16} on {}'.format(name, j['summary']))
            st = func(folder, stats=stats_orig, force=force)
            stats.update(st)
        db.job_update(j['summary'], {'stats': stats})


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



def condgen(*, 
            generator='mnist', 
            discriminator='digits_and_letters', 
            layer_name='output', 
            unit_id='0', 
            nb_iter=500, 
            nb_samples=1, 
            eps1=1., 
            eps2=1.0, 
            eps3=0., 
            out='out.png',
            out_npz='gen.npz',
            discr_loss='log_proba'):
   
    ae = load('ae/results/{}'.format(generator))
    discr = load('discr/{}'.format(discriminator))

    if unit_id == 'all':
        nb_classes = discr.output_shape[1]
        unit_id = np.arange(nb_classes)
    else:
        unit_id = unit_id.split(',')
        unit_id  = map(int, unit_id)
        unit_id = list(unit_id)
        unit_id = np.array(unit_id)

    x = discr.layers[0].input
    xo = x
    x = ae(x)
    y = discr(x)
    if discr_loss == 'log_proba':
        loss = K.log(y[:, unit_id] + 1e-5).mean()
    elif discr_loss == 'objectness':
        loss = _objectness(y)

    grad = K.gradients(loss, xo)
    get_grad = K.function([xo, K.learning_phase()], grad)
    shape = ae.input_shape[1:]
    X = np.random.uniform(0, 1, size=(nb_samples,) + shape)
    X = X.astype('float32')
    Xlist = []
    for i in range(nb_iter):
        Xn = X + eps3 * np.random.normal(0, 1, size=X.shape)
        rec = (ae.predict(Xn) - Xn)
        X += eps1 * rec + eps2 * get_grad([X, 0])
        X = np.clip(X, 0, 1)
        pr = discr.predict(X)
        objectness = compute_objectness(pr)
        confidence = pr[:, unit_id].mean()
        reconstruction = (rec**2).mean()
        print('confidence : {:.3f}, reconstruction : {:.4f}, objectness : {:.4f}'.format(confidence, reconstruction, objectness))
        Xlist.append(X.copy())
    im = np.array(Xlist)
    shape = im.shape[0], im.shape[1]
    im = im.reshape((im.shape[0] * im.shape[1], im.shape[2], im.shape[3], im.shape[4]))
    im = grid_of_images_default(im, shape=shape)
    imsave(out, im)
    Xfull = np.array(Xlist)
    np.savez(out_npz, X=Xfull)


def _objectness(pr):
    marginal = pr.mean(axis=0, keepdims=True)
    score = pr * K.log((pr / marginal) + 1e-10)
    score = score.sum(axis=1)
    return score.mean()


def sanity():
    db = load_db('ae/.lightjob')
    jobs = db.all_jobs()
    for j in jobs:
        print(j['summary'])
        stats = {}
        if j.get('stats') is not None:
            stats.update(j['stats'])
        if 'attractor' in stats:
            del stats['attractor']
        if 'recons' in stats:
            stats['recons_ratio'] = stats['recons']
            del stats['recons']
        cols = {}
        for k, v in stats.items():
            if ('count' in k) or ('objectness' in k) or ('emnist' in k) or ('max' in k) or ('entropy' in k) or ('mmd' in k) or ('frechet' in k) or ('diversity' in k) or ('entropy' in k):
                cols[k] = v
        for k in cols.keys():
            del stats[k]
        if len(cols):
            stats['metrics'].update(cols)
        if 'hwrt' in stats:
            del stats['hwrt']
        db.job_update(j['summary'], {'stats': stats})


if __name__ == '__main__':
    run([evaluate, condgen, extract, sanity])
