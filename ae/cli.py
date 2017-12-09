from clize import run
import os

import numpy as np
from skimage.io import imsave

from machinedesign.viz import grid_of_images_default

from interface import train as _train
from interface import generate as _generate

import hypers
import jobs

from lightjob.utils import summarize
from lightjob.cli import load_db
from lightjob.db import SUCCESS


base_folder = 'results'

def train(job):
    get_params = getattr(hypers, job)
    t, g = get_params()
    t, g = _set_folder(t, g, job)
    if os.path.exists(t['report']['outdir']):
        print('Repository "{}" exists, I dont override for safety. Delete it if you want to.'.format(t['report']['outdir']))
        return
    _train(t)

def train_job(job):
    db = load_db()
    rng = np.random
    get_params = getattr(jobs, job)
    t, g = get_params(rng)
    content = {'train': t, 'generate': g}
    job_id = summarize(content)
    if db.job_exists_by_summary(job_id):
        print('Job exists.')
        return
    nb = db.safe_add_job(content, sampler=job)
    assert nb == 1
    t, g = _set_folder(t, g, os.path.join('jobs', job_id))
    if os.path.exists(t['report']['outdir']):
        print('Repository "{}" exists, I dont override for safety. Delete it if you want to.'.format(t['report']['outdir']))
        return
    _train(t)
    db.modify_state_of(job_id, SUCCESS)

def generate_job(sampler, *, force=False):
    db = load_db()
    for j in db.jobs_with(sampler=sampler):
        t, g = j['content']['train'], j['content']['generate']
        t, g = _set_folder(t, g, os.path.join('jobs', j['summary']))
        if os.path.exists(g['method']['save_folder']) and not force:
            print('Repository {} exists, I dont override for safety. Delete it if you want to'.format(g['method']['save_folder']))
            continue
        try:
            _generate_from(g)
        except Exception as ex:
            print(ex)
            continue

def generate(job):
    get_params = getattr(hypers, job)
    t, g = get_params()
    t, g = _set_folder(t, g, job)
    _generate_from(g)

def _generate_from(g):
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
    run([train, generate, train_job, generate_job])
