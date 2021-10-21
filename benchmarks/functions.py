import torch
from math import pi
"""
`https://pytorch.org/docs/stable/autograd.html`
関数はクラスで定義して、`forward`と(`backward`)を定義する
"""
# sumとmeanの次元を要確認
# 定義域・次元・最適値・最適解も一緒に持っていたい
# 簡単にベンチマークを取れる枠組み作りをしたい
class BaseFunction(object):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
    
    def grad(self, x):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            self(x).sum().backward()
        return x.grad.detach().clone()

class Ackley(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-30, 30)

    def __call__(self, x):
        """バッチ処理対応予定
        x.shape = [bs, n]
        """
        return (-20 * torch.exp(-0.2*x.pow(2).mean(dim=-1).sqrt()) - torch.exp(torch.cos(2*pi*x).mean(dim=-1))).unsqueeze(-1)

class DixonPrice(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-10, 10)
        self._seq = torch.arange(2, self.n)

    def __call__(self, x):
        return ((x[0]-1).pow(2).sum(dim=-1) + (self._seq * (2*x[1:].pow(2) - x[:-1]).pow(2)).sum(dim=-1)).unsqueeze(-1)

class Griewank(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-600, 600)
        self._sqseq = torch.arange(1, n).sqrt()
    
    def __call__(self, x):
        return (x.pow(2).sum(dim=-1) / 4000 - torch.cos(x/self._sqseq).prod(dim=-1) + 1).unsqueeze(-1)
    
class Infinity(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-1, 1)

    def __call__(self, x):
        return (x.pow(6) * (torch.sin(1/x) + 2)).sum(dim=-1).unsqueeze(-1)
    
class Mishra11(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-10, 10)
    def __call__(self, x):
        return (x.abs().mean(dim=-1) - x.abs().prod(dim=-1).sqrt()).pow(2).unsqueeze(-1)

class Multimodal(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-10, 10)
    
    def __call__(self, x):
        return (x.abs().mean(dim=-1) * x.abs().prod(dim=-1)).unsqueeze(-1)

class Plateau(BaseFunction):
    def __init__(self, n=2, minimum=30, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-5.12, 5.12)
    
    def __call__(self, x):
        return (30 * x.abs().sum(dim=-1)).unsqueeze(-1)

class Qing(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-500, 500)
    
    def __call__(self, x):
        return (x.pow(2) - 1).pow(2).sum(dim=-1).unsqueeze(-1)

class Quintic(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-10, 10)
    
    def __call__(self, x):
        return (x.pow(5) -3 * x.pow(4) + 4 * x.pow(3) + 2 * x.pow(2) - x - 4).abs().sum(dim=-1).unsqueeze(-1)

class Rastringin(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-5.12, 5.12)
    
    def __call__(self, x):
        return (10*self.n + (x.pow(2) - 10 * torch.cos(self.n*pi*x)).sum(dim=-1)).unsqueeze(-1)

class Rosenbrock(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-5, 10)
    
    def __call__(self, x):
        return (100 * (x[1:] - x[:-1].pow(2)).pow(2) + (x[:-1]-1).pow(2)).mean(dim=-1).unsqueeze(-1)

class Schwefel21(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-100, 100)
    
    def __call__(self, x):
        return x.abs().max(dim=-1).unsqueeze(-1)

class Schwefel22(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-100, 100)
    
    def __call__(self, x):
        return (x.abs().sum(dim=-1) + x.abs().prod(dim=-1)).unsqueeze(-1)

class Step(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-100, 100)
    
    def __call__(self, x):
        return x.pow(2).sum(dim=-1).unsqueeze(-1)

class StyblinskiTang(BaseFunction):
    def __init__(self, n=2, minimum=None, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        if self.minimum is None:
            self.minimum = -39.1659 * n
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-5, 5)
    
    def __call__(self, x):
        return ((x.pow(4) - 16 * x.pow(2) + 5 * x).sum(dim=-1) / 2).unsqueeze(-1)

class Trid(BaseFunction):
    def __init__(self, n=2, minimum=None, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        if self.minimum is None:
            self.minimum = -n * (n + 4) * (n - 1) / 6
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-pow(n, 2), pow(n, 2))
    
    def __call__(self, x):
        return ((x - 1).pow(2).sum(dim=-1) - x[1:]*x[:-1].sum(dim=-1)).unsqueeze(-1)

class Sphere(BaseFunction):
    def __init__(self, n=2, minimum=0, bounds=None) -> None:
        super().__init__()
        self.n = n
        self.minimum = minimum
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (-5, 5)
    
    def __call__(self, x):
        return x.pow(2).sum(dim=-1).unsqueeze(-1)
