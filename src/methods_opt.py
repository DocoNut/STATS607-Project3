import numpy as np

def kde(h: float, data: np.ndarray):
    """
    Vectorized Kernel Density Estimator.
    Faster execution using NumPy broadcasting.
    """
    # 
    
    if not isinstance(h, (int, float)):
        raise TypeError("h is not number")
    if h <= 0:
        raise ValueError("bandwidth > 0 required")
    
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        raise ValueError("data is empty")

    def density(y):
        # Convert y to array to allow broadcasting
        # Shape becomes (M, 1)
        y_vec = np.atleast_1d(y)[:, None]
        
        # Broadcasting: (M, 1) - (1, N) -> (M, N) difference matrix
        u = (y_vec - data[None, :]) / h
        
        # Vectorized Gaussian calculation
        kernels = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
        
        # Sum across rows (axis=1) to get density for each y
        result = np.sum(kernels, axis=1) / (n * h)
        
        # Return scalar if input was scalar, else array
        if np.isscalar(y):
            return result[0]
        return result

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
    Vectorized Plug-in Kernel Density Estimator with bias correction.
    """
    if not isinstance(h, (int, float)):
        raise TypeError("h is not number")
    if h <= 0:
        raise ValueError("bandwidth > 0 required")
    
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        raise ValueError("data is empty")

    def density(y):
        # 1. Broadcasting setup
        # y_vec: (M, 1), data: (1, N) -> u: (M, N)
        y_vec = np.atleast_1d(y)[:, None]
        u = (y_vec - data[None, :]) / h
        
        # 2. Vectorized Kernel Calculations
        # Standard Gaussian: (1/sqrt(2pi)) * exp(-0.5 * u^2)
        gauss_term = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
        
        # Second derivative of Gaussian: (u^2 - 1) * Gaussian
        # This formula assumes the standard normal kernel K(u)
        deriv_term = (u**2 - 1) * gauss_term

        # 3. Combine terms (Plug-in formula)
        # Formula: K(u) - 0.5 * K''(u)
        # Note: Both terms are divided by (n*h)
        contributions = (gauss_term - 0.5 * deriv_term) / (n * h)
        
        # 4. Sum across data points
        result = np.sum(contributions, axis=1)

        if np.isscalar(y):
            return result[0]
        return result

    return density

def adaptive_kde(h: float, data: np.ndarray):
    """
    Vectorized Adaptive Kernel Density Estimator with bias correction.
    """
    n = len(data)
    f = kde(h, data)

    # 1. Pre-compute pilot densities once (Key Fix)
    pilot_vals = np.maximum(f(data), 1e-10) 
    G = np.exp(np.sum(np.log(pilot_vals)) / n)
    lam = np.sqrt(G / pilot_vals)  # Local bandwidths for each data point

    def density(y: np.ndarray) -> np.ndarray:
        # 2. Vectorized calculation (No Python loops)
        # diffs shape: (len(y), n)
        diffs = (y[:, None] - data[None, :]) / (lam[None, :] * h)
        kernels = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * diffs**2)
        return np.sum(kernels / (lam[None, :] * h), axis=1) / n

    return density