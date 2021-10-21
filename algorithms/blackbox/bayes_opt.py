from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, Matern,
                                              WhiteKernel)


class BayesOpt(object):
    def __init__(self):
        self.kernel = None
        self.Aquisition = None
    def minimize(self, *args, **kwargs):
        pass
    def maximize(self, *args, kwargs):
        return self.minimize(*args, **kwargs)
