from functools import partial
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from machinedesign.autoencoder.interface import load
import keras.backend as K
from skimage.io import imsave
from machinedesign.viz import grid_of_images_default

sns.set_style('white')

digits = np.load('../../data/digits.npz')
digits = digits['X'] / 255.
data = np.load('../../data/hwrt_thin.npz')
hwrt = data['X'] / 255.
model = load('../results/mnist')

inp = model.layers[0].input
hid_layer = model.get_layer('winner_take_all_channel_1')
hid = hid_layer.output
out = model.layers[-1].output
enc = K.function([inp], hid)
dec = K.function([hid], out)

data = np.load('../results/mnist/gen/generated.npz')
X = data['generated'][0:1000]

dae = load('../results/mnist_dense2')

df = pd.read_csv('../../export/hypers.csv')

def fig1():
    h = enc([X])
    h = h.max(axis=(2, 3))
    vals = h.max(axis=0)
    H = np.zeros((hid_layer.input_shape[1],) + hid_layer.input_shape[1:])
    H[np.arange(H.shape[0]), np.arange(H.shape[0]), 8, 8] = vals
    x  = dec([H])
    im = grid_of_images_default(x, normalize=False, shape=(16, 8))
    imsave('features.png', im)


def fig2():
    d = digits[2:3]
    h = enc([d])
    hm = h.max(axis=(2, 3))
    hm = hm[0]
    f = []
    f.append(d)
    for i in range(len(hm)):
        if hm[i]:
            hh = h.copy()
            hh[:, 0:i] = 0
            hh[:, i+1:] = 0
            x = dec([hh])
            f.append(x)
    f.append(d)
    f = np.concatenate(f, axis=0)
    print(f.shape)
    im = grid_of_images_default(f, normalize=False, shape=(len(f), 1))
    imsave('features_example_raw.png', im)


def fig3():
    x = hwrt[0:20] 
    xr = x
    for _ in range(1):
        xr = dae.predict(xr)
    xl = np.array([x,xr])
    xl = xl.transpose((1, 0, 2, 3, 4))
    xl=xl.reshape((xl.shape[0]*xl.shape[1],) + xl.shape[2:])
    im = grid_of_images_default(xl, shape=(len(x),2))
    imsave('reconstructions_hwrt_dae.png', im)


def fig4():
    x = hwrt[0:20] 
    xr=x
    xl=[x]
    for _ in range(30):
        xr = dae.predict(xr)
        xl.append(xr)
    xl = np.array(xl)
    shape = xl.shape[0], xl.shape[1]
    xl=xl.reshape((xl.shape[0]*xl.shape[1],) + xl.shape[2:])
    im = grid_of_images_default(xl, shape=shape)
    imsave('reconstructions_hwrt_dae_iter.png', im)

def fig5():
    nb_layers = 3
    fig = partial(
        _fig,
        yin='recons_ratio_digits',
        yout='recons_ratio_hwrt',
        ylabel='recRatio($\cdot$)'
    )
    # Bottleneck
    d = df
    d = d[d['nb_layers']==nb_layers]
    d = d[d['sampler']  == 'mnist_capacity']
    d['bottleneck'] = d['bottleneck'].astype(int)
    fig(d, xcol='bottleneck', xlabel='Bottleneck size', out='bottleneck_rec_ratio.png', 
        ascending=False)
    
    # Sparsity
    d = df
    d = d[d['nb_layers']==nb_layers]
    d = d[d['sampler']  == 'mnist_deep_kchannel']
    fig(d, xcol='zero_ratio', xlabel=r'Sparsity rate $\rho$', out='sparsity_rec_ratio.png', 
         ascending=True)
    # Noise
    d = df
    d = d[d['sampler']  == 'mnist_noise']
    d = d[d['nb_layers']==nb_layers]
    d = d[d['noise'] <= 0.5]
    fig(d, xcol='noise', xlabel=r'Salt and pepper corruption probability $p_{corruption}$', 
        out='noise_rec_ratio.png', ascending=True)

    # nb layers
    d = df
    d = d[d['sampler']  == 'mnist_deep']
    d = d[d['stride']==0]
    fig(d, xcol='nb_layers', xlabel='Number of layers', out='nb_layers_rec_ratio.png', 
        ascending=True)


def fig6():
    data = np.load('../results/jobs/f316dedd30ebc5f51c4678f07979436c/gen/generated.npz')
    X = data['generated']
    X = X[0:100]
    im = grid_of_images_default(X, border=10, bordercolor=(0, 0, 0))
    imsave('out_of_class_generator.png', im)
    nb_layers = 3
    fig = partial(
        _fig,
        yin='emnist_digits_count',
        yout='emnist_letters_count',
        ylabel=r'$count(\cdot)$',
        plot_in=False,
    )
    #
    # Bottleneck
    d = df
    d = d[d['nb_layers']==nb_layers]
    d = d[d['sampler']  == 'mnist_capacity']
    d['bottleneck'] = d['bottleneck'].astype(int)
    fig(d, xcol='bottleneck', xlabel='Bottleneck size', out='bottleneck_count.png', 
        ascending=False)
    
    # Sparsity
    d = df
    d = d[d['nb_layers']==nb_layers]
    d = d[d['sampler']  == 'mnist_deep_kchannel']
    fig(d, xcol='zero_ratio', xlabel=r'Sparsity rate $\rho$', out='sparsity_count.png', 
        ascending=True)

    # Noise
    d = df
    d = d[d['sampler']  == 'mnist_noise']
    d = d[d['nb_layers']==nb_layers]
    d = d[d['noise'] <= 0.5]
    fig(d, xcol='noise', xlabel=r'Salt and pepper corruption probability $p_{corruption}$', 
        out='noise_count.png', ascending=True)

    # nb layers
    d = df
    d = d[d['sampler']  == 'mnist_deep']
    d = d[d['stride']==1]
    fig(d, xcol='nb_layers', xlabel='Number of layers', out='nb_layers_count.png', 
        ascending=True)

    # global
    sns.lmplot('recons_ratio_hwrt', 'emnist_letters_count', data=df)
    plt.xlabel('recRatio(out)')
    plt.ylabel('count(out)')
    plt.savefig('recons_count.png')

def _fig(d, xcol, yin, yout, xlabel, ylabel, out, ascending=False, plot_in=True, plot_out=True, kind='line'):
    s = 80
    ls = 'solid'
    lw = 2

    d = d.copy()
    d = d.sort_values(by=xcol, ascending=ascending)
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.linspace(0, 1, len(d))
    l = d[xcol].values
    plt.xticks(x, l)
    if ascending is False:
        plt.xticks(x, l[::-1])
        plt.xlim(1, 0)


    if plot_in:
        if kind == 'line':
            y = d[yin]
            if ascending is False: y = y[::-1]
            plt.plot(x, y, zorder=2, color='blue', linestyle=ls, label='in', lw=lw)
            plt.scatter(x, y, zorder=2, color='blue', linestyle=ls, s=s, label='_nolegend_')
        elif kind == 'factorplot':
            pass

    if plot_out:
        if kind == 'line':
            y = d[yout]
            if ascending is False: y = y[::-1]
            plt.plot(x, y, zorder=2, color='green', linestyle=ls, label='out', lw=lw)
            plt.scatter(x, y, zorder=2, color='green', linestyle=ls, s=s, label='_nologend_')
        elif kind == 'factorplot':
            pass
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(out)
    plt.close(fig)
#fig1()
#fig2()
#fig3()
#fig4()
fig5()
fig6()
