import numpy as np
from random_matrix import *
from scipy.optimize import nnls

def nmf_mu(A, r, max_iter=100, random_state=1,tol = 1e-4):
    """
    Non-negative Matrix Factorization using multiplicative update rules.

    Parameters:
        A (np.ndarray): Input non-negative matrix (m x n)
        r (int): Target rank
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance on reconstruction error
        random_state (int or None): Random seed

    Returns:
        W (np.ndarray): (m x r) basis matrix
        H (np.ndarray): (r x n) coefficient matrix
    """
    np.random.seed(random_state)
    m, n = A.shape
    
    # Initialize W,H
    W = np.random.rand(m, r)
    H = np.random.rand(r, n)

    errors = []
    for _ in range(max_iter):
        # Update H
        H *= (W.T @ A) / (W.T @ W @ H + 1e-10)
        
        # Update W
        W *= (A @ H.T) / (W @ H @ H.T + 1e-10)
        
        # Calculate Errors
        error = np.linalg.norm(A - W @ H, 'fro') / np.linalg.norm(A, 'fro')
        errors.append(error)
        if error < tol:
            break
    return W, H, errors

def nmf_compress_mu(A,r,max_iter = 100,random_state = 1,tol = 1e-4, projection_type = 'gaussian'):

    """ 
    Fei Wang
    Ping Li
    /Users/tomcole/python/stat-6104/papers/NMF with Random Projections.pdf
    """

    eps=1e-10
    d, n = A.shape
    np.random.seed(random_state)

    Rd = get_projection_matrix(projection_type,r,d)
    Rn = get_projection_matrix(projection_type,r,n)

    # Create projected matrices
    Xd = Rd @ A 
    Xn = A @ Rn.T  
    
    # Initialize F and G with random non-negative values
    F = np.abs(np.random.randn(d, r))
    G = np.abs(np.random.randn(n, r))
    
    errors = []
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
        G = np.maximum(G, eps)
        
        # Update F (Equation 3.29)
        G_tilde = Rn @ G
        Xn_G = Xn @ G_tilde
        Xn_G_p, Xn_G_n = split_pos_neg(Xn_G)
        
        G_T_G = G_tilde.T @ G_tilde
        G_T_G_p, G_T_G_n = split_pos_neg(G_T_G)
        
        numerator = Xn_G_p + F @ G_T_G_n + eps
        denominator = Xn_G_n + F @ G_T_G_p + eps
        F *= np.sqrt(numerator / denominator)
        F = np.maximum(F, eps)
        
        # Calculate Errors
        error = np.linalg.norm(A - F @ G.T, 'fro') / np.linalg.norm(A, 'fro')
        errors.append(error)
        if error < tol:
            break

    return F, G.T, errors


def nmf_structured_compress_mu(A, r, max_iter=100, random_state=1, tol=1e-4, power_iter=3, oversampling=10,projection_type = 'gaussian'):
    """ 
    Modified NMF with structured random projections using randomized power iteration
    """
    eps = 1e-10
    d, n = A.shape
    np.random.seed(random_state)

    l = r + oversampling
    
    # Construct omega_left and omega_right internally
    omega_left = get_projection_matrix(projection_type,n,l)
    omega_right = get_projection_matrix(projection_type,d,l)
    
    # Perform RPI
    Qd = randomized_power_iteration(A, r, omega=omega_left, 
                                     power_iter=power_iter, random_state=random_state)
    Xd = Qd.T @ A  # Projected matrix (r × n)
    
    Qn = randomized_power_iteration(A.T, r, omega=omega_right, 
                                     power_iter=power_iter, random_state=random_state)
    Xn = A @ Qn    # Projected matrix (d × r)
  
    # Initialize F and G with random non-negative values
    F = np.abs(np.random.randn(d, r))
    G = np.abs(np.random.randn(n, r))
    
    errors = []
    for _ in range(max_iter):
        # Update G (Equation 3.28)
        F_tilde = Qd.T @ F
        Xd_T_F = Xd.T @ F_tilde
        Xd_T_F_p, Xd_T_F_n = split_pos_neg(Xd_T_F)
        
        F_T_F = F_tilde.T @ F_tilde
        F_T_F_p, F_T_F_n = split_pos_neg(F_T_F)
        
        numerator = Xd_T_F_p + G @ F_T_F_n + eps
        denominator = Xd_T_F_n + G @ F_T_F_p + eps
        G *= np.sqrt(numerator / denominator)
        G = np.maximum(G, eps)
        
        # Update F (Equation 3.29)
        G_tilde = Qn.T @ G
        Xn_G = Xn @ G_tilde
        Xn_G_p, Xn_G_n = split_pos_neg(Xn_G)
        
        G_T_G = G_tilde.T @ G_tilde
        G_T_G_p, G_T_G_n = split_pos_neg(G_T_G)
        
        numerator = Xn_G_p + F @ G_T_G_n + eps
        denominator = Xn_G_n + F @ G_T_G_p + eps
        F *= np.sqrt(numerator / denominator)
        F = np.maximum(F, eps)
        
        error = np.linalg.norm(A - F @ G.T, 'fro') / np.linalg.norm(A, 'fro')
        errors.append(error)
        if error < tol:
            break

    return F, G.T, errors



def nmf_hals(A, r, max_iter=100, tol=1e-4, random_state=None):
    np.random.seed(random_state)
    m, n = A.shape

    # Initialize W and H with non-negative values
    W = np.abs(np.random.rand(m, r))
    H = np.abs(np.random.rand(r, n))
    eps=1e-10
    errors = []
    for _ in range(max_iter):
        # --- Update H (row-wise) ---
        WtW = W.T @ W  
        WtA = W.T @ A  
        for k in range(r):
            numerator = WtA[k, :] - WtW[k, :] @ H + WtW[k, k] * H[k, :]
            denominator = max(WtW[k, k], eps)
            H[k, :] = np.maximum(numerator / denominator, 0)

        # --- Update W (column-wise) ---
        HHt = H @ H.T   
        AHt = A @ H.T  
        for k in range(r):
            numerator = AHt[:, k] - W @ HHt[:, k] + HHt[k, k] * W[:, k]
            denominator = max(HHt[k, k], eps)
            W[:, k] = np.maximum(numerator / denominator, 0)

        # Compute error
        error = np.linalg.norm(A - W @ H, 'fro') / np.linalg.norm(A, 'fro')
        errors.append(error)
        if error < tol:
            break

    return W, H, errors

def nmf_compress_hals(A, r, max_iter=100, tol=1e-4, random_state=None,projection_type = 'gaussian'):
    """
    Compressed Non-negative Matrix Factorization using Hierarchical Alternating Least Squares.

    Parameters:
        A (np.ndarray): Input non-negative matrix (m x n)
        r (int): Target rank
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance on reconstruction error
        random_state (int or None): Random seed

    Returns:
        W (np.ndarray): (m x r) basis matrix
        H (np.ndarray): (r x n) coefficient matrix
        errors (list): List of reconstruction errors
    """
    np.random.seed(random_state)
    m, n = A.shape
    eps = 1e-10

    # Generate random projection matrices
    Rd = get_projection_matrix(projection_type,r,m)
    Rn = get_projection_matrix(projection_type,r,n)

    # Create projected matrices
    A_proj_rows = Rd @ A    # r x n
    A_proj_cols = A @ Rn.T  # m x r

    # Initialize F and G (compressed versions of W and H)
    F = np.abs(np.random.rand(m, r))  # m x r
    G = np.abs(np.random.rand(n, r))  # n x r

    errors = []
    for _ in range(max_iter):
        # Update G (compressed version of H) using projected data
        # Project F to lower dimension
        F_tilde = Rd @ F  # r x r
        # Compute necessary matrices in compressed space
        FtA_proj = F_tilde.T @ A_proj_rows  # r x n
        FtF_tilde = F_tilde.T @ F_tilde  # r x r

        # Update each row of G
        for k in range(r):
            # Numerator and denominator in compressed space
            numerator = FtA_proj[k, :] - FtF_tilde[k, :] @ G.T + FtF_tilde[k, k] * G.T[k, :]
            denominator = max(FtF_tilde[k, k], eps)
            G.T[k, :] = np.maximum(numerator / denominator, 0)

        # Update F (compressed version of W) using projected data
        # Project G to lower dimension
        G_tilde = Rn @ G  # r x r
        # Compute necessary matrices in compressed space
        AprojGt = A_proj_cols @ G_tilde  # m x r
        GtG_tilde = G_tilde.T @ G_tilde  # r x r

        # Update each column of F
        for k in range(r):
            # Numerator and denominator in compressed space
            numerator = AprojGt[:, k] - F @ GtG_tilde[:, k] + GtG_tilde[k, k] * F[:, k]
            denominator = max(GtG_tilde[k, k], eps)
            F[:, k] = np.maximum(numerator / denominator, 0)

        # Compute reconstruction error in original space
        error = np.linalg.norm(A - F @ G.T, 'fro') / np.linalg.norm(A, 'fro')
        errors.append(error)
        if error < tol:
            break

    return F, G.T, errors

def nmf_structured_compress_hals(A, r, max_iter=100, tol=1e-4, random_state=None, power_iter=3, oversampling=10, projection_type='gaussian'):
    """
    Structured Compressed Non-negative Matrix Factorization using Hierarchical Alternating Least Squares.

    Parameters:
        A (np.ndarray): Input non-negative matrix (m x n)
        r (int): Target rank
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance on reconstruction error
        random_state (int or None): Random seed
        power_iter (int): Number of power iterations for randomized SVD
        oversampling (int): Oversampling parameter for randomized SVD
        projection_type (str): Type of projection ('gaussian' or other future types)

    Returns:
        W (np.ndarray): (m x r) basis matrix
        H (np.ndarray): (r x n) coefficient matrix
        errors (list): List of reconstruction errors
    """
    np.random.seed(random_state)
    m, n = A.shape
    eps = 1e-10

    l = r + oversampling  # Effective rank with oversampling

    # Projection Matrices
    omega_left = get_projection_matrix(projection_type,n,l)
    omega_right = get_projection_matrix(projection_type,m,l)

    # Compute orthonormal bases using randomized power iteration
    Qd = randomized_power_iteration(A, r, omega=omega_left, power_iter=power_iter, random_state=random_state)  # m x r
    Qn = randomized_power_iteration(A.T, r, omega=omega_right, power_iter=power_iter, random_state=random_state)  # n x r

    # Project the original matrix A
    A_proj_rows = Qd.T @ A  # r x n (projected rows)
    A_proj_cols = A @ Qn  # m x r (projected columns)

    # Initialize F and G (compressed versions of W and H)
    F = np.abs(np.random.rand(m, r))  # m x r
    G = np.abs(np.random.rand(n, r))  # n x r

    errors = []
    for _ in range(max_iter):
        # Update G (compressed version of H) using projected data
        # Project F to lower dimension
        F_tilde = Qd.T @ F  # r x r
        # Compute necessary matrices in compressed space
        FtA_proj = F_tilde.T @ A_proj_rows  # r x n
        FtF_tilde = F_tilde.T @ F_tilde  # r x r

        # Update each row of G
        for k in range(r):
            numerator = FtA_proj[k, :] - FtF_tilde[k, :] @ G.T + FtF_tilde[k, k] * G.T[k, :]
            denominator = max(FtF_tilde[k, k], eps)
            G.T[k, :] = np.maximum(numerator / denominator, 0)

        # Update F (compressed version of W) using projected data
        # Project G to lower dimension
        G_tilde = Qn.T @ G  # r x r
        # Compute necessary matrices in compressed space
        AprojGt = A_proj_cols @ G_tilde  # m x r
        GtG_tilde = G_tilde.T @ G_tilde  # r x r

        # Update each column of F
        for k in range(r):
            numerator = AprojGt[:, k] - F @ GtG_tilde[:, k] + GtG_tilde[k, k] * F[:, k]
            denominator = max(GtG_tilde[k, k], eps)
            F[:, k] = np.maximum(numerator / denominator, 0)

        # Compute reconstruction error in original space
        error = np.linalg.norm(A - F @ G.T, 'fro') / np.linalg.norm(A, 'fro')
        errors.append(error)
        if error < tol:
            break

    return F, G.T, errors