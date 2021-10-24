import numpy as np
import torch
import time
from algorithms.base import BaseOptimizer
from projection import identity
from utils import getInitialPoint

np.random.seed(0)
torch.random.manual_seed(0)

@torch.inference_mode()
class GWO(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, dimension, objective, max_iter, num_population=100,
                 top_k=3, proj=identity, *args, **kwargs):
        self.init_params()
        s = time.time()
        x = getInitialPoint((num_population, dimension), objective)
        objectives = objective(x)
        lis = torch.argsort(objectives.view(-1))
        best_x = x[lis[:top_k]].clone()

        a = torch.full(size=(dimension,), fill_value=2.0)
        r1 = torch.from_numpy(np.random.random(dimension))
        C = 2*torch.from_numpy(np.random.random(dimension))
        A = 2*a*r1-a

        X_s = best_x.clone()
        A_s = A.broadcast_to(X_s.shape)
        C_s = C.broadcast_to(X_s.shape)

        self.gather_info(objectives, x)
        for _ in range(max_iter):
            prod = C_s*X_s
            tmp = 0.
            for i in range(top_k):
                tmp += X_s[i] - (A_s[i]*(prod[i] - x)).abs()
            x = tmp / top_k
            x = proj(x)
            a -= a/max_iter
            r1 = torch.from_numpy(np.random.random(dimension))
            A = 2*a*r1-a
            C = 2*torch.from_numpy(np.random.random(dimension))
            objectives = objective(x)
            lis = torch.argsort(objectives.view(-1))
            best_x = x[lis[:top_k]].clone()
            X_s = best_x.clone()
            A_s = A.broadcast_to(X_s.shape)
            C_s = C.broadcast_to(X_s.shape)
            self.gather_info(objectives, x)
        return self.best_objective, time.time() - s, self.best_x, self.visited_points