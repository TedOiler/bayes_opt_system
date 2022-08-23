import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf


def bellcurve_2d(x_ax, y_ax):
    z = 4 * np.exp(-x_ax ** 2) + 6 * np.exp(-4 * y_ax ** 2)
    return torch.tensor(z.reshape((x_ax.shape[0], 1)))


def get_initial_data(n=10, low=None, high=None):
    if high is None:
        high = [5, 5]
    if low is None:
        low = [-5, -5]
    x = torch.FloatTensor(n, 1).uniform_(low[0], high[0])
    y = torch.FloatTensor(n, 1).uniform_(low[1], high[1])
    z = bellcurve_2d(x, y)
    best_z = z.max().item()
    return x, y, z, best_z


def gen_next_points(X, y, best_y, bounds, n_exp):
    model = SingleTaskGP(X, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    EI = qExpectedImprovement(model=model, best_f=best_y)
    candidates, _ = optimize_acqf(acq_function=EI,
                                  bounds=bounds,
                                  q=n_exp,
                                  num_restarts=200,
                                  raw_samples=1024,
                                  options={"batch_limit": 5, "maxiter": 200})
    return candidates
