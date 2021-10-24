import numpy as np
import torch

def getInitialPoint(shape, objective, initialize_method="random", *args, **kwargs):
    if initialize_method == "random":
        try:
            points = torch.from_numpy(np.random.uniform(*objective.bounds, size=shape))
        except AttributeError:
            points = torch.from_numpy(np.random.random(size=shape))
    else:
        points = torch.zeros(size=shape)
    return points


def preprocess(C, inequalities):
    """add slack variables to transform normal formulation

    Parameters
    ----------
    C : np.array or torch.tensor
        original constraint matrix
    inequalities : list[int]
        制約の不等式の種類を表す。
        -1 : <=
        0 : =
        1 : >=
    
    Returns
    -------
    C : np.array or torch.tensor
        スラック制約を考慮して再構成された制約行列
    """
    n, _ = C.shape
    for i, eq in inequalities:
        if eq == 0:
            continue
        add_vec = torch.zeros(size=(n, 1))
        add_vec[i] = eq
        C = torch.stack([C, add_vec])
    return C
"""考えるべきこと
ランク落ちしている場合の処理

"""