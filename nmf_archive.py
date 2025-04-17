import numpy as np
from random_matrix import *
from scipy.optimize import nnls
from nmf import *

def nmf_compress_nnls(A, r, max_iter=100, random_state=1,tol = 1e-4):

    eps=1e-10
    d, n = A.shape
    np.random.seed(random_state)

    Rd = gaussian_random_matrix(r,d,seed=random_state)
    Rn = gaussian_random_matrix(r,n,seed=random_state)

    # Create projected matrices
    Xd = Rd @ A 
    Xn = A @ Rn.T  
    
    # Initialize F and G with random non-negative values
    F = np.abs(np.random.randn(d, r))
    G = np.abs(np.random.randn(n, r))
    
    
    errors = []
    for _ in range(max_iter):
        # Update G: argmin_{G≥0} ||RdX - RdF G^T||_F^2
        RdF = Rd @ F  # (r x r)
        RdX = Rd @ A  # (r x n)
        
        for j in range(n):
            # Solve min ||RdF @ g_j - RdX[:,j]||^2 s.t. g_j ≥ 0
            G[j], _ = nnls(RdF, RdX[:, j])
        
        # Update F: argmin_{F≥0} ||RnX^T - RnG F^T||_F^2
        RnG = Rn @ G  # (r x r)
        RnXT = Rn @ A.T  # (r x d)
        
        for i in range(d):
            # Solve min ||RnG @ f_i - RnXT[:,i]||^2 s.t. f_i ≥ 0
            F[i], _ = nnls(RnG, RnXT[:, i])
        
        error = np.linalg.norm(A - F @ G.T,'fro') / np.linalg.norm(A, 'fro')
        errors.append(error)
        if error < tol:
            break
    return F,G.T,errors


def nmf_nnls(A, r, max_iter=100, random_state=1,tol = 1e-4):
    """
    Non-negative Matrix Factorization using NNLS (non-negative least squares),
    without early stopping.

    Parameters:
        A (np.ndarray): Input non-negative matrix (m x n)
        r (int): Target rank
        max_iter (int): Number of iterations to run
        random_state (int or None): Random seed

    Returns:
        W (np.ndarray): (m x r) basis matrix
        H (np.ndarray): (r x n) coefficient matrix
    """
    np.random.seed(random_state)
    m, n = A.shape

    # Initialize W,H
    W = np.abs(np.random.rand(m, r))
    H = np.abs(np.random.rand(r, n))

    errors = []
    for _ in range(max_iter):
        # Update H
        for j in range(n):
            H[:, j], _ = nnls(W, A[:, j])
        
        # Update W
        for i in range(m):
            W[i, :], _ = nnls(H.T, A[i, :])
        
        error = np.linalg.norm(A - W @ H, 'fro') / np.linalg.norm(A, 'fro')
        errors.append(error)
        if error < tol:
            break
    return W, H, errors

def nmf_structured_compress_nnls(A, r, max_iter=100, random_state=1, tol=1e-4, 
                               power_iter=3, oversampling=10, projection_type='gaussian'):
    """ 
    NMF with structured random projections using randomized power iteration
    and Non-Negative Least Squares (NNLS) updates
    
    Parameters:
    -----------
    A : ndarray
        Input matrix (d × n)
    r : int
        Target rank
    max_iter : int
        Maximum iterations
    random_state : int
        Random seed
    tol : float
        Convergence tolerance
    power_iter : int
        Number of power iterations
    oversampling : int
        Oversampling factor (rank + oversampling)
    projection_type : str
        Type of random projection ("gaussian", "sparse", "rademacher")
    """
    d, n = A.shape
    np.random.seed(random_state)

    l = r + oversampling
    
    # Construct omega_left and omega_right internally
    if projection_type == "gaussian":
        omega_left = np.random.randn(n, l)
        omega_right = np.random.randn(d, l)
    
    # Create structured random projections
    Qd = randomized_power_iteration(A, r, omega=omega_left, 
                                  power_iter=power_iter, random_state=random_state)
    Rd = Qd.T  # Projection matrix (r × d)
    Xd = Rd @ A  # Projected matrix (r × n)
    
    Qn = randomized_power_iteration(A.T, r, omega=omega_right, 
                                  power_iter=power_iter, random_state=random_state)
    Rn = Qn.T  # Projection matrix (r × n)
    Xn = A @ Rn.T  # Projected matrix (d × r)
    
    # Initialize F and G with random non-negative values
    F = np.abs(np.random.randn(d, r))
    G = np.abs(np.random.randn(n, r))
    
    errors = []
    for _ in range(max_iter):
        # Update G: argmin_{G≥0} ||RdA - RdF G^T||_F^2
        RdF = Rd @ F  # (r × r)
        for j in range(n):
            G[j], _ = nnls(RdF, Xd[:, j],tol=1e-3)  # Xd = RdA
        
        # Update F: argmin_{F≥0} ||RnA^T - RnG F^T||_F^2
        RnG = Rn @ G  # (r × r)
        for i in range(d):
            F[i], _ = nnls(RnG, Xn[i, :].T,tol=1e-3)  # Xn = ARn.T
        
        error = np.linalg.norm(A - F @ G.T, 'fro') / np.linalg.norm(A, 'fro')
        errors.append(error)
        if error < tol:
            break

    return F, G.T, errors