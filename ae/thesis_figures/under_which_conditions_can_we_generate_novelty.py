import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from machinedesign.autoencoder.interface import load
import keras.backend as K
from skimage.io import imsave
from machinedesign.viz import grid_of_images_default
from hypers import get_df

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
    imsave('features_example.png', im)


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
    d = df
    d = d[d['nb_layers']==3]
    d = d[d['sampler']  == 'mnist_capacity']
    d['Bottleneck'] = d['bottleneck'].astype(int)
    d['recRatio'] = d['recons_hwrt_thin']
    d = d.sort_values(by='Bottleneck', ascending=False)
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(
        x='Bottleneck', 
        y='recRatio', 
        data=d, 
        color='blue', 
        edgecolor='black', 
        linewidth=2, 
        order=d['Bottleneck'].values,
    )
    plt.ylabel('recRatio($D_{out}}$)')
    plt.savefig('bottleneck_rec_ratio.png')
    plt.close(fig)

#fig1()
#fig2()
#fig3()
#fig4()
fig5()
