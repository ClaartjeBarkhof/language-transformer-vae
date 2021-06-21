from torch.distributions import MultivariateNormal
import torch


# -------------------------------------------------------------------------------------------
# PyTorch Gaussian KDE (used for optimisation in loss_and_optimisation.py)
# -------------------------------------------------------------------------------------------

# The code for the GaussianKDE class below is taken from this forum post:
# https://discuss.pytorch.org/t/kernel-density-estimation-as-loss-function/62261/8
class GaussianKDE:
    def __init__(self, X, bw, device="cuda:0"):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """

        self.X = X
        # D.W. Scott, “Multivariate Density Estimation: Theory, Practice, and Visualization”, J
        # ohn Wiley & Sons, New York, Chicester, 1992.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        if bw == "scott":
            n, d = X.shape
            bw = n ** (-1. / (d + 4))
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims).to(device),
                                      covariance_matrix=torch.eye(self.dims).to(device))

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.


        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X is None:
            X = self.X

        log_probs = torch.log(
            (self.bw ** (-self.dims) *
             torch.exp(self.mvn.log_prob(
                 (X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n)

        return log_probs