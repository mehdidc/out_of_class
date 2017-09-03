from clize import run

from sklearn.metrics import classification_report

from machinepredict.interface import train as _train
from machinepredict.interface import load

import hypers

from machinedesign.data import pipeline_load

def train(name):
    params = getattr(hypers, name)()
    return _train(params)


def report(name):
    params = getattr(hypers, name)()
    folder = params['report']['outdir']
    clf = load(folder)
    data = pipeline_load(params['data']['test']['pipeline'])
    for d in data:
        X, y = d['X'], d['y']
        y = y.argmax(axis=1)
        y_pred = clf.predict(X).argmax(axis=1)
        print(classification_report(y, y_pred))
 

if __name__ == '__main__':
    run([train, report])
