import numpy as np
from scipy.linalg import sqrtm


def gaussian_kernel(A, B, sigmas=[1]):
    A_ = A[:, None, :]
    B_ = B[None, :, :]
    out = 0
    for sigma in sigmas:
        m = ((A_ - B_) ** 2).sum(axis=2)
        out += np.exp(-m / (2 * sigma**2))
    return out.sum()


def compute_mmd(X, Y, kernel=gaussian_kernel):
    # https://arxiv.org/pdf/1502.02761.pdf
    phi = kernel
    a = phi(X, X)
    b = phi(X, Y)
    c = phi(Y, Y)
    N = X.shape[0]
    M = Y.shape[0]
    mmd_sqr = (1. / (N**2)) * a - (2. / (N * M)) * b + (1. / M**2) * c
    return mmd_sqr
 

def compute_frechet(X, Y):
    # https://arxiv.org/pdf/1706.08500.pdf
    X = X.reshape((X.shape[0], -1))
    Y = Y.reshape((Y.shape[0], -1))

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    cov_x = np.cov(X.T)
    cov_y = np.cov(Y.T)

    return ((mu_x - mu_y)**2).sum() + np.trace(cov_x + cov_y -  2 * sqrtm(np.dot(cov_x, cov_y)))


def compute_objectness(probas):
    # http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf
    pr = probas
    marginal = pr.mean(axis=0, keepdims=True)
    score = pr * np.log((pr / marginal) + 1e-10)
    score = score.sum(axis=1)
    return np.exp(score.mean())


def compute_normalized_entropy(probas):
    pr = probas
    score = -pr * np.log(pr + 1e-10)
    score = score.sum(axis=1)
    return score.mean() / np.log(probas.shape[1])

def compute_normalized_diversity(probas):
    y = probas.argmax(axis=1)
    ent = 0.0
    for cl in range(probas.shape[1]):
        pr = (y == cl).mean()
        ent += -pr * np.log(pr + 1e-10)
    return ent / np.log(probas.shape[1])


def compute_count(probas, classes, theta=0.9):
    pr = probas[:, classes]
    pr = pr.sum(axis=1)
    return (pr > theta).mean()

def compute_max(probas, classes, theta=0.9):
    pr = probas[:, classes]
    pr = pr.max(axis=1)
    return (pr > theta).mean()
