import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from machinedesign.autoencoder.interface import load
import keras.backend as K
from skimage.io import imsave
from machinedesign.viz import grid_of_images_default
from hypers import get_df

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

df = get_df()

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

    # Bottleneck
    d = df
    d = d[d['nb_layers']==nb_layers]
    d = d[d['sampler']  == 'mnist_capacity']
    d['bottleneck'] = d['bottleneck'].astype(int)
    _fig(d, 'bottleneck', 'Bottleneck size', 'bottleneck_rec_ratio.png', ascending=False)
    
    # Sparsity
    d = df
    d = d[d['nb_layers']==nb_layers]
    d = d[d['sampler']  == 'mnist_deep_kchannel']
    _fig(d, 'zero_ratio', 'Sparsity rate', 'sparsity_rec_ratio.png', ascending=True)

    # Noise
    d = df
    d = d[d['sampler']  == 'mnist_noise']
    d = d[d['nb_layers']==nb_layers]
    """
    m = df
    m = m[m['sampler'] == 'mnist_deep']
    m = m[m['stride'] == 0]
    m = m[m['nb_layers'] == nb_layers]
    print(len(m))
    a = m.iloc[0]['recons_digits']
    b = m.iloc[0]['recons_hwrt_thin']
    print(a, b)
    d = d.append(pd.DataFrame({'noise': [0.0], 'recons_digits': [a], 'recons_hwrt_thin': [b]}))
    """
    _fig(d, 'noise', 'Salt-and-pepper noise probability', 'noise_rec_ratio.png', ascending=True)


def _fig(d, col, xlabel, out, ascending=False):
    d = d.copy()
    d['rec_ratio_in'] = d['recons_digits']
    d['rec_ratio_out'] = d['recons_hwrt_thin']
    r = []
    for i in range(len(d)):
        vals = d.iloc[i]
        cols = {}
        cols[col] = vals[col]
        cols['rec_ratio'] = vals['recons_digits']
        cols['type'] = 'in'
        r.append(cols)
        cols = {}
        cols[col] = vals['bottleneck']
        cols['rec_ratio'] = vals['recons_hwrt_thin']
        cols['type'] = 'out'
        r.append(cols)
    dd = pd.DataFrame(r)
    d = d.sort_values(by=col, ascending=ascending)
    fig = plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        x=col, 
        y='rec_ratio', 
        hue='type',
        data=dd, 
        edgecolor=['black'] * len(d),
        linewidth=2,
        order=d[col].values,
    )
    for c in ax.get_children():
        if hasattr(c, 'get_width'):
            width = c.get_width()
            break
    s = 15
    ls = 'dashed'
    plt.plot(ax.get_xticks()-width/2, d['rec_ratio_in'], zorder=2, color='blue', label='_nolegend_', linestyle=ls)
    plt.scatter(ax.get_xticks()-width/2, d['rec_ratio_in'], zorder=2, color='blue', label='_nolegend_', s=s)
    plt.plot(ax.get_xticks()+width-width/2, d['rec_ratio_out'], zorder=2, color='green', label='_nolegend_', linestyle=ls)
    plt.scatter(ax.get_xticks()+width-width/2, d['rec_ratio_out'], zorder=2, color='green', label='_nolegend_',s=s)
    plt.xlabel(xlabel)
    plt.ylabel('recRatio($\cdot$)')
    plt.legend()
    plt.savefig(out)
    plt.close(fig)

#fig1()
#fig2()
#fig3()
#fig4()
fig5()
