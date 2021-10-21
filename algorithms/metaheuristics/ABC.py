"""
Sample implementation of Artificial Bee Colony Algorithm.

Reference : https://link.springer.com/content/pdf/10.1007/s10898-007-9149-x.pdf
"""
import torch
import numpy as np
import time
from algorithms.base import BaseOptimizer
from utils import getInitialPoint
torch.random.manual_seed(0)
np.random.seed(0)

@torch.inference_mode()
class ABC(BaseOptimizer):
    """バッチ処理可能な関数を想定。
    基底クラスに欲しいもの：
    + best_obj
    + best_solution
    + all visited points
    + computation time
    + 経過情報更新関数
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, dimension, objective, max_iter, max_visit=10, num_population=100, *args, **kwargs):
        s = time.time()
        # step1 : initialization
        x = getInitialPoint((num_population, dimension), objective)
        all_candidates = torch.arange(num_population)
        v = objective(x)
        cnt = torch.zeros(num_population)
        m = v.min()

        self.gather_info(v, x, m)

        def update(i):
            x_i = x[i].clone()
            j = np.random.randint(0, dimension-1)
            k = np.random.randint(0, num_population-1)
            phi = np.random.normal()
            x_i[j] -= phi*(x_i[j] - x[k][j])
            v_new = objective(x_i)
            if v_new <= v[i]:
                x[i] = x_i
                v[i] = v_new
            cnt[i] += 1
    
        def random_update():
            candidate = torch.where(cnt == max_visit)[0]
            for i in candidate:
                x_i = getInitialPoint((dimension, ), objective)
                v_new = objective(x_i)
                if v_new <= v[i]:
                    x[i] = x_i
                    v[i] = v_new
                    cnt[i] = 1
    
        for _ in range(1, max_iter+1):
            for _ in range(num_population):
                # employed bees
                i = np.random.randint(0, num_population-1)
                update(i)
    
                # onlooker bees
                # 確率のスケーリング方法は諸説あり
                if (v >= 0).all():
                    probs = v / v.sum()
                else:
                    m = v.min()
                    w = v - m
                    probs = w / w.sum()
                probs = (1 - probs).squeeze(1)
                probs /= probs.sum()
                i = np.random.choice(all_candidates, p=probs)
                update(i)
    
                # scouts
                random_update()
            m = v.min()
            self.gather_info(v, x, m)
    
        return self.best_objective, time.time() - s, self.best_x, self.visited_points
