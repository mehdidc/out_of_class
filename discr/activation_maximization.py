import os

from skimage.io import imsave
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from clize import run

from machinedesign.data import floatX
from machinedesign.viz import grid_of_images_default
from machinepredict.interface import load
from machinepredict.interface import predict

from keras import backend as K

def main():
    params = {
        'model':{
            'folder': 'digits_and_letters'
        },
        'nb_examples': 10,
        'nb_iter': 200,
        'layer_id': -2,
        'unit_id': 1,
        'learning_rate': 0.5,
        'regularization': [
            {
                'name': 'blur',
                'params':{
                    'sigma': 0.7,
                }
            },
            {
                'name': 'l2',
                'params':{
                    'decay': 0.01,
                }
            }
        ],
        'save_folder': 'digits_and_letters/gen',
    }
    activation_maximization(params)
    data = np.load('{}/activation_maximization.npz'.format(params['save_folder']))
    X = data['full']
    X = X.transpose((1, 0, 2, 3, 4))
    nb_iter, nb_examples = X.shape[0], X.shape[1]
    X = X.reshape((nb_iter * nb_examples,) + X.shape[2:])
    img = grid_of_images_default(X, shape=(nb_iter, nb_examples))
    imsave('{}/samples.png'.format(params['save_folder']), img)



def activation_maximization(params):
    folder = params['model']['folder']
    nb_examples = params['nb_examples']
    nb_iter = params['nb_iter']
    layer_id = params['layer_id']
    unit_id = params['unit_id']
    save_folder = params['save_folder']
    reg = params['regularization']
    lr = params['learning_rate']
    out_filename = os.path.join(save_folder, 'activation_maximization.npz')

    model = load(folder)
    inp_layer = model.layers[0]
    x = K.placeholder(inp_layer.input_shape)
    h = x
    for l in model.layers[0:layer_id + 1]:
        h = l(h)
    loss = h[:, unit_id].mean()
    g = K.gradients(loss, x)
    get_gradients = K.function([x, K.learning_phase()], g)
    get_loss = K.function([x, K.learning_phase()], loss)
    
    X = np.random.uniform(0, 1, size=(nb_examples, nb_iter + 1) + inp_layer.input_shape[1:])

    for i in range(1, nb_iter + 1):
        grads = get_gradients([X[:, i - 1], False])
        grads = normalize(grads)
        X[:, i] = X[:, i - 1] + lr * grads
        for r in reg:
            X[:, i] = apply_reg(r, X[:, i], epoch=i)
        #print(X.min(), X.max())
        val = float(get_loss([X[:, i], False]))
        #pred = model.predict(X[:, i])
        #print(pred[0])
        print('Iteration : {:05d}, Loss : {:.4f}'.format(i, val))
    np.savez(out_filename, full=X, generated=X[:, -1])


def apply_reg(reg, X, epoch=None):
    name = reg['name']
    params = reg['params']
    if name == 'l2':
        decay = params['decay']
        return (1.0 - decay) * X
    elif name == 'clip':
        minval = params['min']
        maxval = params['max']
        return np.clip(X, minval, maxval)
    elif name == 'blur':
        sigma = params['sigma']
        for i in range(X.shape[0]):
            X[i] = gaussian_filter(X[i], sigma)
        return X
    elif name == 'none':
        return X
    else:
        raise ValueError('Unknown regularization method "{}"'.format(name))
    
def normalize(x):
    return x / (np.sqrt(np.mean(x**2) + 1e-5))

if __name__ == '__main__':
    run(main)
