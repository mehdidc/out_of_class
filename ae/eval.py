import os
from clize import run
from functools import lru_cache
import numpy as np
from keras.models import Model
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
    datasets = ['hwrt_thin', 'digits_test']
    names = ['hwrt', 'digits']
    if all([n in stat for n in names]) and not force:
        print('skip')
        return {}
    theta = 50 
    nb = 10000
    out = {}
    model = load(folder)
    for name, ds in zip(names, datasets):
        data = np.load('../data/{}.npz'.format(ds))
        X = data['X'][0:nb] / 255.0
        Xr = model.predict(X)
        err = (np.abs(X-Xr).sum(axis=(1, 2, 3)) < theta).mean()
        err = float(err)
        out[name] = err
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

    data = np.load('../data/digits.npz')
    Xtrue_digits = data['X'][0:nb]
    data = np.load('../data/letters.npz')
    Xtrue_letters = data['X'][0:nb]

    clf_digits_and_letters = load('../discr/digits_and_letters_balanced')
    clf_digits = load('../discr/digits')
    clf_letters = load('../discr/letters')
    clf_hwrt = load('../discr/hwrt')
    clf_emnist = load('../discr/emnist')

    enc = Model(
        inputs=clf_digits_and_letters.layers[0].input, 
        outputs=clf_digits_and_letters.get_layer('p_re_lu_4').output
    )
    htrue_digits = enc.predict(Xtrue_digits)
    htrue_letters = enc.predict(Xtrue_letters)
    if not os.path.exists(os.path.join(folder, 'model.h5')):
        print('model does not exist')
        return {}
    col = {}
    model = load(folder)
    col['nb_params'] = model.count_params()
    filename = '{}/gen/generated.npz'.format(folder)
    if not os.path.exists(filename):
        print('gen does not exist')
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


def evaluate(*, force=False, name=None, job=None, sampler=None):
    kw = {}
    if sampler:
        kw['sampler'] = sampler
    if job:
        kw['summary'] = job
    db = load_db()
    jobs = db.jobs_with(**kw)
    if name is None:
        funcs = eval_funcs.items()
    else:
        funcs = (name, eval_funcs[name]),

    for j in jobs:
        folder = os.path.join('results', 'jobs', j['summary'])
        stats = {}
        if j.get('stats') is not None:
            stats.update(j['stats'])
        stats_orig = stats.copy() 
        for name, func in funcs:
            print('Eval of {:<16} on {}'.format(name, j['summary']))
            st = func(folder, stats=stats_orig, force=force)
            stats.update(st)
        print(stats)
        db.job_update(j['summary'], {'stats': stats})


def sanity():
    db = load_db()
    jobs = db.all_jobs()
    """
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
    """
    for j in jobs:
        stats = j['stats']
        stats['recons_ratio']['hwrt'] = stats['recons_ratio']['hwrt_thin']
        del stats['recons_ratio']['hwrt_thin']
        stats['recons_ratio']['digits'] = stats['recons_ratio']['digits_test']
        del stats['recons_ratio']['digits_test']
        print(stats['recons_ratio'])
        db.job_update(j['summary'], {'stats': stats})


if __name__ == '__main__':
    run([evaluate, sanity])
