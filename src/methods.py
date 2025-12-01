import numpy as np
from scipy.stats import f, norm

def kernel_gaussian(z: float) -> float:
    """
    Standard Gaussian kernel.

    Args:
        z (float): Input value.

    Returns:
        float: Kernel value at z.
    """
    return np.exp(-z * z / 2) / np.sqrt(2 * np.pi)


def kernel_derivative(z: float) -> float:
    """
    First derivative of the Gaussian kernel.

    Args:
        z (float): Input value.

    Returns:
        float: Derivative value at z.
    """
    return -z * kernel_gaussian(z)


def kernel_second_derivative(z: float) -> float:
    """
    Second derivative of the Gaussian kernel.

    Args:
        z (float): Input value.

    Returns:
        float: Second derivative value at z.
    """
    return (z * z - 1) * kernel_gaussian(z)


def kde(h: float, data: np.ndarray):
    """
    Construct a kernel density estimator (KDE) function with Gaussian kernel.

    Args:
        h (float): Bandwidth parameter.
        data (np.ndarray): 1D sample points.

    Returns:
        callable: Function f(y) that estimates density at y.
    """
    if not isinstance(h, (int, float)):
        raise TypeError("h is not number")
    
    if not isinstance(data, np.ndarray):
        raise TypeError("data is not np.ndarray")
    
    if h <= 0:
        raise ValueError("bandwidth should be greater than 0")
    
    n = len(data)
    if n < 0:
        raise ValueError("data is empty")

    def density(y: float) -> float:
        # Sum contributions from each data point
        total = 0.0
        for xi in data:
            total += kernel_gaussian((y - xi) / h) / (n * h)
        return total

    return density


def multi_kde(h: float, data: np.ndarray, bandwidth_coef: np.ndarray, d: int = 2, lam: float = 0):
    """
    Construct a multi-bandwidth KDE (linear combination of KDEs at different scales).

    Args:
        h (float): Base bandwidth multiplier.
        data (np.ndarray): 1D sample points.
        bandwidths (np.ndarray): Array of scale factors for bandwidths.
        d (int): Polynomial degree for coefficient calculation.
        lam (float): Regularization parameter.

    Returns:
        callable: Function f(y) that estimates density at y.
    """
    if not isinstance(h, (int, float)):
        raise TypeError("h is not number")
    
    if not isinstance(data, np.ndarray):
        raise TypeError("data is not np.ndarray")
    
    if h <= 0:
        raise ValueError("bandwidth should be greater than 0")
    
    n = len(data)
    if n < 0:
        raise ValueError("data is empty")
    
    from src.metrics import coef  # imported here to avoid circular import

    def density(y: float) -> float:
        # Compute coefficients for combining KDEs
        coeffs = coef(bandwidth_coef, degree=d, lam=lam)

        # Weighted sum of individual KDEs at scaled bandwidths
        total = 0.0
        for i in range(len(bandwidth_coef)):
            total += coeffs[i] * kde(bandwidth_coef[i] * h, data)(y)
        return total

    return density


def multi_kde_n0(h: float, data: np.ndarray, bandwidths: np.ndarray, d: int = 2, lam: float = 0):
    """
    Multi-bandwidth KDE that enforces non-negativity (clipped at 0).

    Args:
        h (float): Base bandwidth multiplier.
        data (np.ndarray): 1D sample points.
        bandwidths (np.ndarray): Array of scale factors for bandwidths.
        d (int): Polynomial degree for coefficient calculation.
        lam (float): Regularization parameter.

    Returns:
        callable: Function f(y) that estimates density at y (≥ 0).
    """
    def density(y: float) -> float:
        # Compute density but clip negative values to 0
        return np.maximum(multi_kde(h, data, bandwidths, d, lam)(y), 0.0)

    return density

def plugin_kde(h: float, data: np.ndarray):
    """
    Construct a plug-in kernel density estimator (KDE) with bias correction.

    Args:
        h (float): Bandwidth parameter.
        data (np.ndarray): 1D array of sample points.

    Returns:
        callable: Function f(y) that estimates density at y.
    """
    if not isinstance(h, (int, float)):
        raise TypeError("h is not number")
    
    if not isinstance(data, np.ndarray):
        raise TypeError("data is not np.ndarray")
    
    if h <= 0:
        raise ValueError("bandwidth should be greater than 0")
    
    n = len(data)
    if n < 0:
        raise ValueError("data is empty")

    def density(y: float) -> float:
        total = 0.0
        for xi in data:
            # Standard Gaussian KDE term minus a second-derivative correction
            total += (
                kernel_gaussian((y - xi) / h) / (n * h)
                - 0.5 * kernel_second_derivative((y - xi) / h) / (n * h)
            )
        return total

    return density


def adaptive_kde(h: float, data: np.ndarray):
    """
    Construct an adaptive kernel density estimator (KDE),
    where the bandwidth varies depending on local density.

    Args:
        h (float): Initial bandwidth parameter.
        data (np.ndarray): 1D array of sample points.

    Returns:
        callable: Function f(y) that estimates density at y.
    """
    n = len(data)

    # Standard KDE for global density estimation
    f = kde(h, data)

    # Global density factor (geometric mean of densities)
    G = np.exp(np.sum(np.log(f(data))) / n)

    def density(y: np.ndarray) -> np.ndarray:
        total = np.zeros(len(y))
        for xi in data:
            # Local bandwidth scaling factor
            lam = np.sqrt(G / f(xi))
            # Contribution from each data point
            total += kernel_gaussian((y - xi) / (lam * h)) / (h * lam)
        return total / n

    return density


def k_nearest_density(data: np.ndarray, ratio: float):
    """
    Construct a density estimator based on the k-nearest neighbor method.

    Args:
        data (np.ndarray): 1D array of sample points.
        ratio (float): Ratio for determining k (k = int(ratio * n)).

    Returns:
        callable: Function f(y) that estimates density at points y.
    """
    if not isinstance(ratio, (int, float)):
        raise TypeError("h is not number")
    
    if not isinstance(data, np.ndarray):
        raise TypeError("data is not np.ndarray")
    
    if ratio <= 0 or ratio >= 1:
        raise ValueError("ratio should be between 0 and 1")

    data = np.sort(data)
    n = len(data)
    if n < 0:
        raise ValueError("data is empty")
    
    k = max(int(np.floor(ratio * n)), 1)  # Ensure k ≥ 1

    def density(y: np.ndarray) -> np.ndarray:
        """
        Estimate density at each point in y using k-nearest neighbors.

        Args:
            y (np.ndarray): Points at which to estimate density.

        Returns:
            np.ndarray: Estimated densities at each point in y.
        """
        p = np.zeros_like(y, dtype=float)
        for i, point in enumerate(y):
            dists = np.abs(data - point)
            neighbor_idx = np.argsort(dists)
            neighbor_k = neighbor_idx[k]  # index of k-th nearest neighbor
            p[i] = (k / n) / (2 * dists[neighbor_k])
        return p

    return density


def bimodal_normal_density(mu1: float, sigma1: float, mu2: float, sigma2: float, p: float):
    """
    Construct a probability density function (PDF) for a bimodal Gaussian mixture.

    Args:
        mu1 (float): Mean of the first Gaussian component.
        sigma1 (float): Standard deviation of the first Gaussian component.
        mu2 (float): Mean of the second Gaussian component.
        sigma2 (float): Standard deviation of the second Gaussian component.
        p (float): Mixing weight for the first component (0 ≤ p ≤ 1).

    Returns:
        callable: Function f(x) that gives the mixture density at x.
    """
    def density(x: np.ndarray) -> np.ndarray:
        return p * norm.pdf(x, mu1, sigma1) + (1 - p) * norm.pdf(x, mu2, sigma2)

    return density