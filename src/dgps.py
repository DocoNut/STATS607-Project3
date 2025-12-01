import numpy as np
from scipy.stats import f as f_dist, beta as beta_dist, norm as norm_dist
import matplotlib.pyplot as plt

class DistributionSampler:
    """
    A unified sampler for various distributions.

    Initialization:
        dist_type (str): Type of the distribution. One of:
            'normal'  -> params: [mean, sd]
            'f'       -> params: [dfn, dfd]
            'beta'    -> params: [d1, d2]
            'bimodal' -> params: [mu1, mu2, s1, s2, p]
        params (list of floats): Distribution-specific parameters (see above).

    Methods:
        _generate_samples(sample_size)
            Generate `sample_size` iid samples from the specified distribution.
        pdf(x)
            Compute the true pdf of the distribution at x.
    """

    def __init__(self, dist_type, params, seed = 1234):
        # Store distribution type (normalized to lowercase) and its parameters
        self.dist_type = dist_type.lower()
        self.params = params

        if self.dist_type == 'bimodal':
            if len(self.params) != 5:
                raise ValueError(f'bimodal distribution requires 5 parameters, you gave {len(self.params)}')
        elif self.dist_type in ('f', 'normal', 'beta'):
            if len(self.params) != 2:
                raise ValueError(f'{self.dist_type} distribution requires 2 parameters, you gave {len(self.params)}')
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")
        # Initialize the pdf function based on type and params
        self.pdf = self.get_pdf()
        np.random.seed(seed)

    def generate_samples(self, sample_size):
        """
        sample_size (int): Number of iid samples to generate.
        Returns:
            np.ndarray of length `sample_size`.
        """
        if self.dist_type == 'normal':
            # params = [mean, sd]
            mean, sd = self.params
            return np.random.normal(mean, sd, sample_size)
        elif self.dist_type == 'f':
            # params = [dfn, dfd]
            dfn, dfd = self.params
            return np.random.f(dfn, dfd, sample_size)
        elif self.dist_type == 'beta':
            # params = [d1, d2]
            d1, d2 = self.params
            return np.random.beta(d1, d2, sample_size)
        elif self.dist_type == 'bimodal':
            # params = [mu1, mu2, s1, s2, p]
            mu1, mu2, s1, s2, p = self.params
            n1 = int(sample_size * p)
            n2 = sample_size - n1
            samples1 = np.random.normal(mu1, s1, n1)
            samples2 = np.random.normal(mu2, s2, n2)
            return np.concatenate((samples1, samples2))

    def get_pdf(self):
        """
        Returns:
            A function that maps x -> pdf(x) for the specified distribution.
        """
        if self.dist_type == 'normal':
            mean, sd = self.params
            return lambda x: norm_dist.pdf(x, mean, sd)
        elif self.dist_type == 'f':
            dfn, dfd = self.params
            return lambda x: f_dist.pdf(x, dfn, dfd)
        elif self.dist_type == 'beta':
            d1, d2 = self.params
            return lambda x: beta_dist.pdf(x, d1, d2)
        elif self.dist_type == 'bimodal':
            mu1, mu2, s1, s2, p = self.params
            return lambda x: p * norm_dist.pdf(x, mu1, s1) + (1 - p) * norm_dist.pdf(x, mu2, s2)