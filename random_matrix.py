import numpy as np
from scipy import sparse
from scipy.linalg import hadamard, dft
from numpy.random import rand, permutation

def randomized_power_iteration(A, rank, omega=None, power_iter=3, random_state=None):
    """
    Randomized subspace iteration with customizable projection matrix
    
    Parameters:
    -----------
    A : ndarray
        Input matrix to approximate
    rank : int
        Target rank of approximation
    omega : ndarray, optional
        Custom projection matrix (shape n × (rank + oversampling))
        If None, uses Gaussian random matrix
    power_iter : int
        Number of power iterations
    random_state : int, optional
        Random seed for reproducibility
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = A.shape[1]
    if omega is None:
        # Default to Gaussian random matrix if none provided
        omega = np.random.randn(n, rank)
    
    Y = A @ omega
    for _ in range(power_iter):
        Y = A @ (A.T @ Y)
    
    Q, _ = np.linalg.qr(Y)
    return Q

def gaussian_random_matrix(n, l, seed=1):
    """
    Generates an n x l matrix with i.i.d. Gaussian entries.
    If normalize=True, scales by 1/sqrt(l) for JL embedding.
    """
    np.random.seed(seed)
    G = np.random.randn(n, l)
    return G / np.sqrt(l)

"""
def srht_matrix(n, l, seed=1):
    
    Generates an n x l SRHT matrix:
    1. Random sign flip (diagonal D).
    2. Hadamard transform (H).
    3. Uniform column sampling (S).
    
    np.random.seed(seed)
    # Pad n to next power of 2 for Hadamard
    n_pad = 2 ** int(np.ceil(np.log2(n)))
    
    # Diagonal matrix D (random ±1)
    D = np.diag(np.random.choice([-1, 1], size=n_pad))
    
    # Hadamard matrix H
    H = hadamard(n_pad)
    
    # Uniform column sampling S
    cols = np.random.choice(n_pad, size=l, replace=True)
    S = np.zeros((n_pad, l))
    S[cols, np.arange(l)] = 1
    
    # Combine: SRHT = DHS (trimmed to n rows)
    SRHT = (D @ H @ S)[:n, :] * np.sqrt(n_pad / l)  # Scaling
    return SRHT
"""

def srht_matrix(n, l, seed=1):
    """
    Generates an n x l SRHT matrix efficiently:
    1. Random sign flip (diagonal D).
    2. Hadamard transform (H, applied implicitly).
    3. Uniform column sampling (S).
    """
    np.random.seed(seed)
    
    # Pad n to next power of 2
    n_pad = 2 ** int(np.ceil(np.log2(n)))
    
    # Random signs for D
    signs = np.random.choice([-1, 1], size=n_pad)
    
    # Uniform column sampling
    cols = np.random.choice(n_pad, size=l, replace=True)
    
    # Compute SRHT = D @ H @ S efficiently
    # S selects l columns, so H @ S is columns of H corresponding to 'cols'
    SRHT = np.zeros((n_pad, l))
    H = hadamard(n_pad)  # Still using hadamard for simplicity
    for i, col in enumerate(cols):
        SRHT[:, i] = H[:, col]  # Select column col of H
    
    # Apply D (element-wise multiplication by signs)
    SRHT = signs[:, np.newaxis] * SRHT
    
    # Trim to n rows and scale
    SRHT = SRHT[:n, :] * np.sqrt(n_pad / l)
    return SRHT

"""
def srft_matrix(n, l, seed=1):

    Generates an n x l SRFT matrix:
    1. Random phase diagonal (D).
    2. DFT matrix (F).
    3. Uniform column sampling (S).

    np.random.seed(seed)
    # Diagonal matrix D (random unit complex numbers)
    D = np.diag(np.exp(1j * 2 * np.pi * np.random.rand(n)))
    
    # DFT matrix F (unitary)
    F = np.fft.fft(np.eye(n), norm="ortho")
    
    # Uniform column sampling S
    cols = np.random.choice(n, size=l, replace=True)
    S = np.zeros((n, l), dtype=np.float64)
    S[cols, np.arange(l)] = 1
    
    # Combine: SRFT = DFS (scaled)
    SRFT = (D @ F @ S).real * np.sqrt(n / l)
    return SRFT
"""

def srft_matrix(n, l, seed=1):
    """
    Generates an n x l SRFT matrix efficiently:
    1. Random phase diagonal (D).
    2. DFT matrix (F, applied implicitly).
    3. Uniform column sampling (S).
    """
    np.random.seed(seed)
    
    # Random phases for D (diagonal entries)
    phases = np.exp(1j * 2 * np.pi * np.random.rand(n))
    
    # Uniform column sampling (select l columns)
    cols = np.random.choice(n, size=l, replace=True)
    
    # Compute SRFT = D @ F @ S efficiently
    # S selects l columns, so F @ S is columns of F corresponding to 'cols'
    # Instead of forming F, compute FFT on the standard basis vectors e_j for j in cols
    SRFT = np.zeros((n, l), dtype=np.complex128)
    for i, col in enumerate(cols):
        # Compute FFT of the standard basis vector e_col
        e_col = np.zeros(n, dtype=np.complex128)
        e_col[col] = 1
        SRFT[:, i] = np.fft.fft(e_col, norm="ortho")
    
    # Apply D (element-wise multiplication of rows by phases)
    SRFT = phases[:, np.newaxis] * SRFT
    
    # Take real part and scale
    SRFT = SRFT.real * np.sqrt(n / l)
    return SRFT

def countsketch_matrix(n, l, seed=1):
    """
    Generates an n x l CountSketch matrix:
    - Exactly 1 nonzero per column (uniform row, random ±1).
    """
    np.random.seed(seed)
    rows = np.random.randint(0, n, size=l)
    cols = np.arange(l)
    vals = np.random.choice([-1, 1], size=l)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, l))

def sparse_jl_matrix(n, l, s=3, seed=1):
    """
    Generates an n x l Sparse JL matrix:
    - s nonzeros per column (random positions, values ±1/sqrt(s)).
    """
    np.random.seed(seed)
    rows = np.random.randint(0, n, size=(s, l))
    cols = np.arange(l).reshape(1, -1).repeat(s, axis=0)
    vals = np.random.choice([-1, 1], size=(s, l)) / np.sqrt(s)
    return sparse.csr_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(n, l))

def haar_orthonormal_matrix(n, l,seed=1):
    """Generate an n × l Haar-distributed orthonormal matrix."""
    np.random.seed(1)
    Z = np.random.randn(n, l)  # Gaussian random matrix
    Q, R = np.linalg.qr(Z)     # QR decomposition
    D = np.diag(np.sign(np.diag(R)))  # Fix sign to ensure uniformity
    return Q @ D


def random_givens_rotation(dim, i, j, theta):
    """Constructs a Givens rotation G(i,j;θ) in C^dim."""
    G = np.eye(dim, dtype=complex)
    G[i, i] = np.cos(theta)
    G[i, j] = -np.sin(theta)
    G[j, i] = np.sin(theta)
    G[j, j] = np.cos(theta)
    return G

def random_givens_chain(n, k=10):
    """Builds Θ with only k random Givens rotations (instead of n-1)."""
    Theta = np.eye(n)
    for _ in range(k):
        i, j = np.random.choice(n, 2, replace=False)  # Random pair of indices
        theta = 2 * np.pi * np.random.rand()          # Random angle
        G = random_givens_rotation(n, i, j, theta)    # Single rotation
        Theta = Theta @ G                             # Apply to chain
    return Theta

def givens_matrix(m, n,seed = 1):
    """Constructs rectangular Ω = D'' Θ' D' Θ D F R."""
    np.random.seed(seed)
    # Random diagonal matrices (D: n×n, D': m×m, D'': m×m)
    D = np.diag(np.sign(np.random.randn(n)))  # Efficient ±1 diagonal
    D_prime = np.diag(np.sign(np.random.randn(m)))
    D_dprime = np.diag(np.sign(np.random.randn(m)))
    
    # Givens chains (Θ: n×n, Θ': m×m)
    Theta = random_givens_chain(n)
    Theta_prime = random_givens_chain(m)
    
    # DFT matrix (n×n) and subsampling matrix R (n×n → m×n)
    F = np.fft.fft(np.eye(n)) / np.sqrt(n)  # Normalized DFT via FFT
    R = np.eye(n)[:m, :] if m < n else np.hstack([np.eye(m), np.zeros((m, n-m))])
    
    # Build Ω = D'' Θ' D' Θ D F R
    Omega = D_dprime @ Theta_prime @ D_prime @ Theta[:m, :n] @ D @ F @ R.T
    return Omega.real


def generate_synthetic_matrix(m, r = 20, delta=0.01, n=None,seed = 1):
    """
    Generates synthetic matrix A = X_GT @ Y_GT + N, where sparsity is controlled by `delta`.
    Always uses dense computation (ignores sparse optimizations).

    Args:
        m:      Rows in A (and X_GT).
        r:      Rank of decomposition.
        delta:  Probability of non-zero entries in X_GT/Y_GT (0 = fully sparse, 1 = fully dense).
        noise:  If True, adds sparse noise (density = delta²).
        n:      Columns in A (and Y_GT). If None, n = int(0.75 * m).

    Returns:
        A:      Synthetic matrix (m × n).
        X_GT:   Ground truth left factor (m × r).
        Y_GT:   Ground truth right factor (r × n).
    """
    if n is None:
        n = int(0.75 * m)
    
    np.random.seed(1)
    # Generate X_GT (m × r) with sparsity delta
    mask_x = np.random.rand(m, r) < delta
    X_GT = np.random.uniform(0, 1, (m, r)) * mask_x
    
    # Generate Y_GT (r × n) with sparsity delta
    mask_y = np.random.rand(r, n) < delta
    Y_GT = np.random.uniform(0, 1, (r, n)) * mask_y
    
    # Generate noise matrix 
    N = np.zeros((m, n))
    mask_n = np.random.rand(m, n) < (delta ** 2)
    N = np.random.randn(m, n) * mask_n
    
    # Compute A
    A = X_GT @ Y_GT + N
    
    return A, X_GT, Y_GT


def split_pos_neg(M):
        return np.maximum(M, 0), np.maximum(-M, 0)

def get_projection_matrix(projection_type,m,n):

    if projection_type == 'gaussian':
        return gaussian_random_matrix(m,n)
    if projection_type == 'srht':
        return srht_matrix(m,n)
    if projection_type == 'srft':
        return srft_matrix(m,n)
    if projection_type == 'sparse-jl':
        return sparse_jl_matrix(m,n)
    if projection_type == 'count-sketch':
        return countsketch_matrix(m,n)
    if projection_type == 'givens':
        return givens_matrix(m,n)

