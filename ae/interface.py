from machinedesign.autoencoder.interface import train as _train
from machinedesign.autoencoder.interface import generate as _generate
from machinedesign.autoencoder.interface import load as _load
from machinedesign.autoencoder.interface import default_config

from model_builders import builders

config = default_config
config.model_builders.update(builders)

def train(params):
    return _train(params, config=config)

def generate(params):
    return _generate(params)

def load(folder):
    return _load(folder)
