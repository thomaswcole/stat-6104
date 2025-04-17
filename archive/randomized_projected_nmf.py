import numpy as np
from scipy.optimize import nnls
from scipy.sparse.linalg import svds
from random_matrix import * 

def nmf_random_projections_as(X, r, k1=None, k2=None, max_iter=100, random_state=None, eps=1e-10,projection = 'gaussian'):
    """
    Non-negative Matrix Factorization with Random Projections
    
    Parameters:
    - X: Input data matrix (d x n)
    - r: Number of components
    - k1: Projection dimension for rows (d dimension). If None, set to r
    - k2: Projection dimension for columns (n dimension). If None, set to r
    - max_iter: Number of iterations
    - random_state: Random seed for reproducibility
    - eps: Small constant to prevent division by zero
    
    Returns:
    - F: Basis matrix (d x r)
    - G: Coefficient matrix (n x r)
    """
    d, n = X.shape
    k1 = r if k1 is None else k1
    k2 = r if k2 is None else k2
    
    # Set random seed
    np.random.seed(random_state)
    
    # Create random projection matrices
    if projection == 'gaussian':
        Rd = gaussian_random_matrix(k1,d)
        Rn = gaussian_random_matrix(k2,n)
    elif projection == 'srht':
        Rd = srht_matrix(k1,d)
        Rn = srht_matrix(k2,n)
    elif projection == 'srft':
        Rd = srft_matrix(k1,d)
        Rn = srft_matrix(k2,n)   
    elif projection == 'countsketch':
        Rd = countsketch_matrix(k1,d)
        Rn = countsketch_matrix(k2,n)
    elif projection == 'sparsejl':
        Rd = sparse_jl_matrix(k1,d)
        Rn = sparse_jl_matrix(k2,n)
    # Create projected matrices
    Xd = Rd @ X  # (k1 x n)
    Xn = X @ Rn.T  # (d x k2)
    
    # Initialize F and G with random non-negative values
    F = np.abs(np.random.randn(d, r))
    G = np.abs(np.random.randn(n, r))
    
    # Helper function to split matrix into positive and negative parts
    def split_pos_neg(M):
        return np.maximum(M, 0), np.maximum(-M, 0)
    
    errors = []
    error = np.linalg.norm(X - F @ G.T,'fro')
    errors.append(error)
    for _ in range(max_iter):
        # Update G (Equation 3.28)
        F_tilde = Rd @ F
        Xd_T_F = Xd.T @ F_tilde
        Xd_T_F_p, Xd_T_F_n = split_pos_neg(Xd_T_F)
        
        F_T_F = F_tilde.T @ F_tilde
        F_T_F_p, F_T_F_n = split_pos_neg(F_T_F)
        
        numerator = Xd_T_F_p + G @ F_T_F_n + eps
        denominator = Xd_T_F_n + G @ F_T_F_p + eps
        G *= np.sqrt(numerator / denominator)
        G = np.maximum(G, eps)  # Ensure non-negativity
        
        # Update F (Equation 3.29)
        G_tilde = Rn @ G
        Xn_G = Xn @ G_tilde
        Xn_G_p, Xn_G_n = split_pos_neg(Xn_G)
        
        G_T_G = G_tilde.T @ G_tilde
        G_T_G_p, G_T_G_n = split_pos_neg(G_T_G)
        
        numerator = Xn_G_p + F @ G_T_G_n + eps
        denominator = Xn_G_n + F @ G_T_G_p + eps
        F *= np.sqrt(numerator / denominator)
        F = np.maximum(F, eps)  # Ensure non-negativity
        
        error = np.linalg.norm(X - F @ G.T,'fro')
        errors.append(error)
        
    return F, G, errors

def nmf_random_projections_mu(X, r, k1=None, k2=None, max_iter=100, random_state=None):
    """
    NMF with Random Projections using exact formulation from the paper
    
    Parameters:
    - X: Input data matrix (d x n)
    - r: Number of components
    - k1: Row projection dimension (default = r)
    - k2: Column projection dimension (default = r)
    - max_iter: Outer iterations
    - random_state: Random seed
    
    Returns:
    - F: Basis matrix (d x r)
    - G: Coefficient matrix (n x r)
    """
    d, n = X.shape
    k1 = r if k1 is None else k1
    k2 = r if k2 is None else k2
    
    np.random.seed(random_state)
    
    # Create random projection matrices with proper scaling
    Rd = np.random.randn(k1, d) / np.sqrt(k1)  # (k1 x d)
    Rn = np.random.randn(k2, n) / np.sqrt(k2)  # (k2 x n)
    
    # Initialize F and G with random non-negative values
    U, S, Vt = svds(X, k=r)
    F = np.abs(U @ np.diag(np.sqrt(S)))  # d×r
    G = np.abs(Vt.T @ np.diag(np.sqrt(S)))  # n×r
    # F = np.abs(np.random.randn(d, r))
    # G = np.abs(np.random.randn(n, r))
    
    errors = []
    error = np.linalg.norm(X - F @ G.T,'fro')
    errors.append(error)
    for _ in range(max_iter):
        # Update G: argmin_{G≥0} ||RdX - RdF G^T||_F^2
        # Equivalent to solving NNLS for each column of G^T (row of G)
        RdF = Rd @ F  # (k1 x r)
        RdX = Rd @ X  # (k1 x n)
        
        for j in range(n):
            # Solve min ||RdF @ g_j - RdX[:,j]||^2 s.t. g_j ≥ 0
            G[j], _ = nnls(RdF, RdX[:, j])
        
        # Update F: argmin_{F≥0} ||RnX^T - RnG F^T||_F^2
        # Equivalent to solving NNLS for each column of F^T (row of F)
        RnG = Rn @ G  # (k2 x r)
        RnXT = Rn @ X.T  # (k2 x d)
        
        for i in range(d):
            # Solve min ||RnG @ f_i - RnXT[:,i]||^2 s.t. f_i ≥ 0
            F[i], _ = nnls(RnG, RnXT[:, i])
        
        error = np.linalg.norm(X - F @ G.T,'fro')
        errors.append(error)
    
    return F, G, errors