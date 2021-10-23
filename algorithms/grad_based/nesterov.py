from math import sqrt
import time
from algorithms.base import GradOptimizer
from utils import getInitialPoint

class NesterovAcceralation(GradOptimizer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, dimension, objective, max_iter, step=1e-4, method="armijo", n_start=10, *args, **kwargs):
        self.init_params()
        s = time.time()
        x = getInitialPoint((n_start, dimension,), objective)
        lam = 1
        lam_nx = None
        gam = -1
        y = x.detach().clone()

        d_prev = -objective.grad(x).detach()
        self.gather_info(objective(x), x)
        for t in range(max_iter):
            y_nx = x + d_prev*alpha
            x = y_nx + gam*(y_nx - y)
            d = -objective.grad(x).detach()

            y = y_nx
            lam_nx = 1 + sqrt(1+2*lam**2)/2
            gam = (lam - 1)/lam_nx
            lam = lam_nx

            alpha = self.getStep((1+gam)*x - gam*y, (1+gam) *
                            d, objective, step=step, method=method, *args, **kwargs)
            d_prev = d.detach().clone()
            self.gather_info(objective(x), x)
        return self.best_objective, time.time() - s, self.best_x, self.visited_points