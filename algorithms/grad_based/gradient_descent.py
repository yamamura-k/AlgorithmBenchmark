import time
from algorithms.base import GradOptimizer
from utils import getInitialPoint
from projection import identity


class GradientDescent(GradOptimizer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, dimension, objective, max_iter, step=1e-4,
                method="armijo", n_start=10, proj=identity, *args, **kwargs):
        self.init_params()
        s = time.time()
        x = getInitialPoint((n_start, dimension,), objective)
        self.gather_info(objective(x), x)

        for t in range(max_iter):
            d = -proj(objective.grad(x).detach())
            alpha = self.getStep(x, d, objective,
                            step=step, method=method, *args, **kwargs)
            x += alpha*d
            self.gather_info(objective(x), x)
        
        return self.best_objective, time.time() - s, self.best_x, self.visited_points
