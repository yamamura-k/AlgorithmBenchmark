"""
References :
- https://ja.wikipedia.org/wiki/%E5%85%B1%E5%BD%B9%E5%8B%BE%E9%85%8D%E6%B3%95
- https://ja.wikipedia.org/wiki/%E9%9D%9E%E7%B7%9A%E5%BD%A2%E5%85%B1%E5%BD%B9%E5%8B%BE%E9%85%8D%E6%B3%95
- 基礎数学 IV 最適化理論
"""
import time
import torch
from algorithms.base import GradOptimizer
from utils import getInitialPoint


class ConjugateGradientDescent(GradOptimizer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, dimension, objective, max_iter, n_start=10, method="armijo", beta_method="default", *args, **kwargs):
        self.init_params()
        stime = time.time()
        x = getInitialPoint((n_start, dimension,), objective)
        d = -objective.grad(x).detach()
        d_prev = d
        s = d
        alpha = self.getStep(x, s, objective, method=method, *args, **kwargs)
        self.gather_info(objective(x), x)
        for t in range(max_iter):
            x += alpha*s
            d = -objective.grad(x).detach()
            beta = self.getBeta(beta_method, d, d_prev, s)
            s = beta*s + d
            alpha = self.getStep(x, s, objective, method=method, *args, **kwargs)
            d_prev = d
            self.gather_info(objective(x), x)
        return self.best_objective, time.time() - stime, self.best_x, self.visited_points

    def getBeta(self, method, d, d_prev, s):
        y = (d - d_prev)
        if method == "default":
            beta = self.getBeta("PR", d, d_prev, s)
            beta[beta < 0] = 0.
            return beta
        elif method == "FR":
            return ((d * d).sum(-1) / (d_prev * d_prev).sum(-1)).unsqueeze(1)
        elif method == "PR":
            return ((d * y).sum(-1) / (d_prev * d_prev).sum(-1)).unsqueeze(1)
        elif method == "HS":
            return ((-d * y).sum(-1) / (s * y).sum(-1)).unsqueeze(1)
        elif method == "DY":
            return ((-d * d).sum(-1) / (s * y).sum(-1)).unsqueeze(1)
        elif method == "HZ":
            return ((y - 2 * (s * (y * y).sum(-1).unsqueeze(1) / (s * y).sum(-1).unsqueeze(1))) * d).sum(-1).unsqueeze(1) / (s * y).sum(-1).unsqueeze(1)
        elif method == "DL":
            return (((y - s) * d).sum(-1) / (s * y).sum(-1)).unsqueeze(1)
        elif method == "LS":
            return (-(d * y).sum(-1) / (s * d_prev).sum(-1)).unsqueeze(1)
        else:
            raise NotImplementedError