from machinepredict.interface import load
import numpy as np

def compute_objectness(probas):
    pr = probas
    marginal = pr.mean(axis=0, keepdims=True)
    score = pr * np.log(pr / marginal)
    score = score.sum(axis=1)
    return score.mean()



def main():
    clf = load('discr/hwrt')
    X = np.load('../ae_gen/mnist/generated.npz')['X']
    X = X[0]
    probas = clf.predict(X)
    print(compute_objectness(probas))


main()
