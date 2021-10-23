import time
from algorithms.base import GradOptimizer
from utils import getInitialPoint


class GradientDescent(GradOptimizer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, dimension, objective, max_iter, step=1e-4,
                method="armijo", n_start=10, *args, **kwargs):
        self.init_params()
        s = time.time()
        x = getInitialPoint((n_start, dimension,), objective)
        self.gather_info(objective(x), x)

        for t in range(max_iter):
            alpha = self.getStep(x, objective.grad(x).detach(), objective,
                            step=step, method=method, *args, **kwargs)
            d = objective.grad(x).detach()
            x += alpha*d
            self.gather_info(objective(x), x)
        
        return self.best_objective, time.time() - s, self.best_x, self.visited_points
