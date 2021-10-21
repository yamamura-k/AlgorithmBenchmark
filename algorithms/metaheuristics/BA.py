import numpy as np
import torch
from utils import getInitialPoint
from algorithms.base import BaseOptimizer, INF
import time
np.random.seed(0)
torch.random.manual_seed(0)
# numpy version

@torch.inference_mode()
class BA(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, dimension, objective, max_iter, num_population=100, f_min=0,
                 f_max=100, selection_max=10, alpha=0.9, gamma=0.9, *args, **kwargs):
        s = time.time()
        x = getInitialPoint((num_population, dimension), objective)
        v = torch.from_numpy(np.random.random((num_population, dimension)))
        f = torch.from_numpy(np.random.uniform(f_min, f_max, size=num_population))
        A = torch.from_numpy(np.random.uniform(1, 2, size=num_population))
        r = torch.from_numpy(np.random.uniform(0, 1, size=num_population))
        r0 = r.clone()
        obj_current = objective(x)
        self.gather_info(obj_current, x)

        for step in range(max_iter):
            f = f_min + (f_max - f_min) * \
                np.broadcast_to(np.random.uniform(
                    0, 1, size=num_population), (dimension, num_population)).T
            v += (x - self.best_x) * f
            x_t = x + v
            obj_t = objective(x_t)
            obj_new = torch.full(size=(num_population, 1), fill_value=INF, dtype=torch.double)
    
            idxs = torch.where(torch.rand(*r.shape) > r)
            x_new = torch.empty_like(x)
            idx = np.random.randint(0, selection_max, size=(len(idxs[0]),))
            eps = torch.from_numpy(np.random.uniform(-1, 1, size=(len(idxs[0]),))).unsqueeze(0).T
            x_new[idxs] = x[idx] + eps * A.mean()
            obj_new[idxs] = objective(x_new[idxs])
    
            x_random = getInitialPoint((num_population, dimension), objective)
            obj_random = objective(x_random)
    
            idxs1 = np.where(((obj_new == np.inf) | (obj_t > obj_new)) & (
                obj_t > obj_random) & (obj_random > obj_new))
            x[idxs1] = x_new[idxs1]
    
            idxs2 = np.where(((obj_new == np.inf) | (obj_t > obj_new)) & (
                obj_t > obj_random) & (~(obj_random > obj_new)))
            x[idxs2] = x_random[idxs2]
    
            idxs3 = np.where(~(((obj_new == np.inf) | (obj_t > obj_new)) & (
                obj_t > obj_random)))
            x[idxs3] = x_t[idxs3]
    
            r = r0 * (1-np.exp(-gamma*step))
            A *= alpha
            obj_current = objective(x)
            self.gather_info(obj_current, x)
        
        return self.best_objective, time.time() - s, self.best_x, self.visited_points