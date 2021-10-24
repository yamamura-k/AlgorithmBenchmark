import numpy as np
import torch
import time
from algorithms.base import BaseOptimizer
from projection import identity
from utils import getInitialPoint

np.random.seed(0)
torch.random.manual_seed(0)

class TLBO(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, dimension, objective, max_iter, num_population=100, proj=identity, *args, **kwargs):
        self.init_params()
        s = time.time()
        x = getInitialPoint((num_population, dimension), objective)
        obj_vals = objective(x)
        all_idx = np.arange(num_population)
        self.gather_info(obj_vals, x)

        for t in range(max_iter):
            teacher = np.argmin(obj_vals)
            mean = x.mean(dim=0)
            Tf = np.round(1+np.random.random())
            r = np.random.random()
            difference_mean = r*(x[teacher] - mean*Tf)
            x_new = proj(x + difference_mean)
            comp_idxs = np.random.choice(all_idx, size=num_population)

            tmp = comp_idxs[np.where(comp_idxs == all_idx)].view()
            tmp = (tmp+1) % num_population

            better_idx = torch.where(obj_vals < obj_vals[comp_idxs])[0]
            other_idx = torch.where(obj_vals >= obj_vals[comp_idxs])[0]
            x_new[better_idx] = x[better_idx] + r * \
                (x[better_idx] - x[comp_idxs[better_idx]])
            x_new[other_idx] = x[other_idx] + r * \
                (x[other_idx] - x[comp_idxs[other_idx]])

            obj_new = objective(x_new)
            update_idxs = torch.where(obj_new < obj_vals)[0]

            x[update_idxs] = proj(x_new[update_idxs])
            obj_vals[update_idxs] = obj_new[update_idxs]
            self.gather_info(obj_vals, x)

        return self.best_objective, time.time() - s, self.best_x, self.visited_points
