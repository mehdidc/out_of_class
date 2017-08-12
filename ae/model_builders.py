from keras.layers import Input
from keras.layers import Average
from keras.models import Model

from machinedesign.common import check_model_shape_or_exception
from machinedesign.common import activation_function
from machinedesign.common import conv2d_layers_all
from machinedesign.common import conv2d_layers
from machinedesign.layers import Convolution2D
from machinedesign.layers import UpConv2D

def vertebrate(params, input_shape, output_shape):
    assert input_shape == output_shape
    nb_channels = input_shape[0]

    stride = params['stride']

    encode_nb_filters = params['encode_nb_filters']
    encode_filter_sizes = params['encode_filter_sizes']
    encode_activations = params['encode_activations']

    code_activations = params['code_activations']

    decode = params['decode']
    output_activation = params['output_activation']

    inp = Input(input_shape)
    x = inp
    hid_layers = conv2d_layers_all(
        x,
        nb_filters=encode_nb_filters,
        filter_sizes=encode_filter_sizes,
        activations=encode_activations,
        border_mode='valid' if stride == 1 else 'same',
        stride=stride,
        conv_layer=Convolution2D,
    )
    # Apply code activations (e.g. sparsity) to all layers in `hid_layers`
    for i, (layer, acts) in enumerate(zip(hid_layers, code_activations)):
        for act in acts:
            layer = activation_function(act)(layer)
        hid_layers[i] = layer
    # Reconstruct from each layer of `hid_layers`
    for i, (layer, decode_params) in enumerate(zip(hid_layers, decode)):
        nb_filters = decode_params['nb_filters']
        filter_sizes = decode_params['filter_sizes']
        activations = decode_params['activations']
        output_filter_size = decode_params['output_filter_size']
        layer = conv2d_layers(
            layer,
            nb_filters=nb_filters + [nb_channels],
            filter_sizes=filter_sizes + [output_filter_size],
            activations=activations + ['linear'],
            border_mode='full' if stride == 1 else 'same',
            stride=stride,
            conv_layer=UpConv2D,
        )
        hid_layers[i] = layer
    # Combine `hid_layers`
    x = Average()(hid_layers)
    x = activation_function(output_activation)(x)
    out = x
    model = Model(inputs=inp, outputs=out)
    check_model_shape_or_exception(model, output_shape)
    return model

builders = {
    'vertebrate': vertebrate,
}
