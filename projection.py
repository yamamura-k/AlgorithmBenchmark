import torch
def compute_orthogonalProjection_matrix(A):
    """compute orthogonal projection of linear constraints

    Parameters
    ----------
    A : np.ndarray or torch.tensor
        constraint matrix
    
    Returns
    -------
    P : np.array or torch.tensor
        orthogonal projection matrix

    Notes
    -----
    Linear constraints means the following type of constraint:
        Ax = b
    We assume that A is a m x n matrix, m < n, and rank(A)=m.
    """
    B = torch.bmm(A , A.T)
    I = torch.eye(B.shape[0])
    # 今回Aのランクが大きくても30程度なので、逆行列計算はコストではない
    P = I - torch.bmm(torch.bmm(A.T, torch.linalg.inv(B)), A)
    return P
class OrthogonalProjection(object):
    def __init__(self, A) -> None:
        super().__init__()
        self.P = compute_orthogonalProjection_matrix(A)
    def __call__(self, d):
        return torch.matmul(self.P, d)

class BoxProjection(object):
    def __init__(self, upper, lower) -> None:
        super().__init__()
        self.upper = upper
        self.lower = lower
    
    def __call__(self, x):
        return torch.clamp(x, min=self.lower, max=self.upper)

def identity(x):
    return x