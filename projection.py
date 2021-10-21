import numpy as np
def orthogonalProjection(A):
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
    B = A @ A.T
    I = np.identity(B.shape[0])
    # 今回Aのランクが大きくても30程度なので、逆行列計算はコストではない
    P = I - A.T @ np.linalg.inv(B) @ A
    return P