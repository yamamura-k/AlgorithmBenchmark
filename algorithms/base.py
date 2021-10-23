from abc import abstractmethod
import torch

INF = float('inf')
class BaseOptimizer(object):
    def __init__(self) -> None:
        super().__init__()
        self.best_objective = INF
        self.best_x = None
        self.visited_points = []
        self.computation_time = None
    
    def init_params(self):
        self.best_objective = INF
        self.best_x = None
        self.visited_points = []
        self.computation_time = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def gather_info(self, objectives, current_points, current_best=None):
        if current_best is None:
            current_best = objectives.min().clone().item()
        if current_best < self.best_objective:
            index, _ = torch.where(objectives == current_best)
            self.best_objective = current_best
            self.best_x = current_points[index].detach().clone()
        self.visited_points.append(current_points.detach().clone())

class GradOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()

    def getStep(self, x, d, objective, step=None, method="armijo", *args, **kwargs):
        if method == "static":
            return step
        elif method == "armijo":
            return self._armijo(x, d, objective, *args, **kwargs)
        elif method == "wolfe":
            return self._wolfe(x, d, objective, *args, **kwargs)

    def _wolfe(self, x, d, objective, c1=1e-5, c2=1-1e-5, alpha=10, *args, **kwargs):
        nab = objective.grad(x)
        f = objective(x)
        _alpha = torch.full_like(f, alpha)
        a = torch.full_like(f, 0)
        b = torch.full_like(f, INF)
        phi_dif0 = (nab*d).sum(-1).unsqueeze(1)
        assert (phi_dif0 <= 0).any()
        prev_alpha = _alpha.clone()
        while True:
            phi = objective(x+_alpha*d)
            phi_dif = (objective.grad(x+_alpha*d) * d).sum(-1).unsqueeze(1)
            condition1 = phi > f + c1 * _alpha * phi_dif0
            condition2 = phi_dif < c2 * phi_dif0
            b[condition1] = _alpha[condition1]
            a[~condition1 & condition2] = alpha[~condition1 & condition2]
            if (~condition1 & ~condition2).all():
                return _alpha
            condition3 = b < INF
            _alpha[condition3] = ((a + b) / 2)[condition3]
            if (_alpha == prev_alpha).all():
                return _alpha
            prev_alpha = _alpha.clone()
            _alpha[~condition3] = 2*a[~condition3]
            prev_alpha = _alpha.clone()
    
    def _armijo(self, x, d, objective, c1=0.5, alpha=10, rho=0.9, *args, **kwargs):
        nab = objective.grad(x)
        f = objective(x)
        _alpha = torch.full_like(f, alpha)
        phi_dif0 = (nab*d).sum(-1).unsqueeze(1)
        if (phi_dif0 >= 0).all():
            return torch.zeros_like(f)
        while True:
            phi = objective(x+_alpha*d)
            condition = (phi > f + c1*_alpha*phi_dif0)
            _alpha[condition] *= rho
            if (~condition).all():
                return _alpha
    
    
    def _check_grad(x, objective):
        n = x.shape[0]
        h = 1e-6
        I = torch.eye(n, n)*h
        x_h = I + x
        x_b = -I + x
        grad = (objective(x_h) - objective(x_b)) / 2 / h
        grad_ = objective.grad(x)
        diff = (grad_ - grad).abs()
    
        assert (diff < 1e-8).all()
