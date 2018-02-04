import os
from clize import run
import keras.backend as K
from skimage.io import imsave
import numpy as np
from machinedesign.viz import grid_of_images_default, grid_of_images
from machinedesign.autoencoder.interface import load
from lightjob.cli import load_db

def thumbnails():
    folders = os.listdir('ae/results/jobs')
    for folder in folders:
        print(folder)
        dirname = os.path.join('ae', 'results', 'jobs', folder)
        filename = os.path.join(dirname, 'gen', 'generated.npz')
        if not os.path.exists(filename):
            continue
        data = np.load(filename)
        X = data['generated']
        X = X[0:25]
        im = grid_of_images_default(X)
        imsave('export/thumbnails/{}.png'.format(folder), im)
    

def extract(*, generator='mnist', discriminator='letters', classes=None, out='extracted.png', nb=10, limit=None):
    data = np.load('ae/results/{}/gen/generated.npz'.format(generator))
    X = data['generated']
    if limit:
        limit = int(limit)
        X = X[0:limit]
    discr = load('discr/{}'.format(discriminator))
    y = discr.predict(X)

    if classes is None:
        classes = np.arange(y.shape[1])
    else:
        classes = classes.split(',')
        classes = [int(c) for c in classes]
        classes = np.array(classes)

    x = []
    for c in classes:
        idx = y[:, c].argsort()[::-1]
        x.append(X[idx][0:nb])
    x = np.array(x)
    x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
    im = grid_of_images(x, shape=(len(classes), nb))
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
    run([condgen, extract, thumbnails])
