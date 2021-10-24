import numpy as np
import torch
import time
from algorithms.base import BaseOptimizer
from projection import identity
from utils import getInitialPoint

np.random.seed(0)
torch.random.manual_seed(0)

@torch.inference_mode()
class FA(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, dimension, objective, max_iter, num_population=100,
                 beta=1, gamma=1, alpha=0.2, proj=identity, *args, **kwargs):
        self.init_params()
        s = time.time()
        x = getInitialPoint((num_population, dimension), objective)
        I = objective(x)
        self.gather_info(I, x)
        for _ in range(max_iter):
            for i in range(num_population):
                # vector implementation
                better_x = x[np.where(I < I[i])[0]].clone()
                better_x -= x[i]
                norm = better_x.pow(2).sum(dim=0)
                rand = torch.from_numpy(np.random.random(size=(better_x.shape[0], 1)))
                x[i] += (beta*torch.exp(-gamma*norm).unsqueeze(0) * better_x +
                            alpha*(rand-0.5)).sum(dim=0)
                x[i] = proj(x[i])
                assert (x[torch.where(I < I[i])[0]] != better_x).all(), f"{I[i].item()}"
                I[i] = objective(x[i].unsqueeze(0))
            self.gather_info(I, x)

        return self.best_objective, time.time() - s, self.best_x, self.visited_points
