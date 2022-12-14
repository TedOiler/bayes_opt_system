import numpy as np
from scipy.stats import norm
import random


def surrogate(model, X):
    return model.predict(X, return_std=True)


def acq_pi(X, Xsamples, model, beta=None):
    yhat, _ = surrogate(model, X)
    best = max(yhat)  # calculate the best so far

    mu, std = surrogate(model, Xsamples)
    mu = mu.reshape(-1, 1)[:, 0]
    gamma = (mu - best) / (std + 1E-9)
    probs = norm.cdf(gamma)  # add a very small number to std to avoid dividing with 0
    return probs


def acq_ei(X, Xsamples, model, beta=None):
    yhat, _ = surrogate(model, X)
    best = max(yhat)  # calculate the best so far

    mu, std = surrogate(model, Xsamples)
    mu = mu.reshape(-1, 1)[:, 0]
    gamma = (mu - best) / (std + 1E-9)  # add a very small number to std to avoid dividing with 0
    probs = std * (gamma * norm.cdf(gamma)) + norm.pdf(gamma)
    return probs


def acq_ucb(X, Xsamples, model, beta):
    yhat, _ = surrogate(model, X)
    mu, std = surrogate(model, Xsamples)
    mu = mu.reshape(-1, 1)[:, 0]
    probs = mu + np.sqrt(beta)*std
    return probs


def opt_acq(X, y, model, acq, low, high, beta):
    Xsamples = get_random_X(low=low, high=high, samples=1000)
    scores = acq(X, Xsamples, model, beta=beta)
    ix = np.argmax(scores)
    return Xsamples[ix]


def get_random_X(low, high, samples):
    ALL = np.array(np.meshgrid(*[np.arange(low[k], high[k], 5) for k in range(len(low))])).T.reshape(-1, len(low))
    idx = random.sample(range(0, ALL.shape[0] - 1), samples)
    return ALL[idx]
