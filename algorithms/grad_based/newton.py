import torch
import time
from torch.autograd import grad
from torch.autograd.functional import hessian
from algorithms.base import GradOptimizer
from utils import getInitialPoint

class Newton(GradOptimizer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, dimension, objective, eps=1e-10, n_start=10, *args, **kwargs):
        self.init_params()
        s = time.time()
        x = getInitialPoint((n_start, dimension,), objective)
        nab = objective.grad(x).detach()
        H_inv = torch.linalg.pinv(objective.hesse(x).detach())
        lam = nab.T@H_inv@nab
        d = -H_inv@nab
        self.gather_info(objective(x), x)
        t = 0
        while lam > eps:
            x = x + d
            nab = objective.grad(x).detach()
            H_inv = torch.linalg.pinv(objective.hesse(x).detach())
            lam = nab.T@H_inv@nab
            d = -H_inv@nab
            t += 1
            self.gather_info(objective(x), x)
        return self.best_objective, time.time() - s, self.best_x, self.visited_points
