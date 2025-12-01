import numpy as np

def invert_symmetric_matrix(A: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a symmetric matrix using eigen decomposition.

    Args:
        A (np.ndarray): Symmetric matrix of shape (n, n).

    Raises:
        ValueError: If the matrix is singular (any eigenvalue = 0).

    Returns:
        np.ndarray: Inverse of A, same shape as A.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("A is not np.ndarray")
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not a square matrix")
    
    # Eigen decomposition (for symmetric matrices, eigh is efficient & stable)
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Check invertibility
    if np.any(eigenvalues == 0):
        raise ValueError("Matrix is singular and cannot be inverted")

    #Compute the inverse matrix
    Lambda_inv = np.diag(1.0 / eigenvalues)
    return eigenvectors @ Lambda_inv @ eigenvectors.T


def cov_gaussian(h1: float, h2: float) -> float:
    """
    Compute the covariance between two Gaussian kernels with different bandwidth
    (Elements of B matrix in report).

    Args:
        h1 (float), h2 (float): Bandwidth.

    Returns:
        float: Covariance.
    """
    return 1 / np.sqrt(2 * np.pi * (h1 * h1 + h2 * h2))


def coef(bandwidth_coef: np.ndarray, degree: int = 2, lam: float = 0) -> np.ndarray:
    """
    Compute coefficient vector for kernel regression approximation.

    Args:
        bandwidth_coef (np.ndarray): 1D array of bandwidth coefficients.
        degree (int): number of constraints on higher order biases.
        lam (float): Regularization parameter (ridge penalty).

    Returns:
        np.ndarray: Coefficient vector (length = len(xi)).
    """
    if not isinstance(bandwidth_coef, np.ndarray):
        raise TypeError("bandwidth are not np.ndarray")

    if not isinstance(degree,int):
        raise TypeError("degree is not integer")

    if not isinstance(lam, (int,float)):
        raise TypeError("lambda is not number")
    
    if degree <= 0:
        raise ValueError("degree should be greater than 0")
    
    if lam < 0:
        raise ValueError("lambda should be non-negative")
    
    n = len(bandwidth_coef)

    # Kernel (Gram) matrix with optional ridge regularization
    B = np.array([[cov_gaussian(x1, x2) for x2 in bandwidth_coef] for x1 in bandwidth_coef]) + lam * np.identity(n)

    # Polynomial feature matrix: columns are [xi^0, xi^2, xi^4, ..., xi^(2*(d-1))]
    X = np.column_stack([bandwidth_coef ** (2 * k) for k in range(degree)])

    # Compute (Xᵀ B⁻¹ X)⁻¹
    M = invert_symmetric_matrix(X.T @ invert_symmetric_matrix(B) @ X)

    # First column of M corresponds to intercept-related part
    coefficients = invert_symmetric_matrix(B) @ X @ M[:, 0]

    return coefficients


def var(bandwidth_coef: np.ndarray, degree: int = 2, lam: float = 0) -> float:
    """
    Compute variance term in kernel regression approximation.

    Args:
        xi (np.ndarray): 1D array of bandwidths or design points.
        d (int): Degree (number of polynomial features).
        lam (float): Regularization parameter (ridge penalty).

    Returns:
        float: Variance coefficient.
    """
    # Kernel (Gram) matrix
    B = np.array([[cov_gaussian(x1, x2) for x2 in bandwidth_coef] for x1 in bandwidth_coef])

    # Coefficient vector
    c = coef(bandwidth_coef,degree, lam)

    # Variance term: cᵀ B c
    return float(c.T @ B @ c)