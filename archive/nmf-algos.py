import numpy as np
from numpy.fft import fft
from scipy.linalg import hadamard

def standard_compression(A, r, rOV, w):
    m, n = A.shape
    d = r + rOV  # Effective reduced dimension

    # Step 1: Draw a Gaussian random matrix Omega_L
    Omega_L = np.random.randn(n, d)

    # Step 2: Compute B = (A A^T)^w A Omega_L
    B = A @ Omega_L  # Initial multiplication: A Omega_L
    for _ in range(w):
        B = (A @ A.T) @ B # Power iteration: (A A^T) B

    # Step 3: Compute the orthogonal basis Q using QR decomposition
    Q, _ = np.linalg.qr(B)

    return Q

def srht_compression(A, r, rOV, w):

    m, n = A.shape
    d = r + rOV  # Effective reduced dimension

    # Step 1: Generate subsampled randomized Hadamard matrix Omega_L
    # Find smallest power of 2 greater than or equal to n
    n_power = 2**int(np.ceil(np.log2(n)))
    
    # Generate Hadamard matrix of size n_power x n_power
    H = hadamard(n_power)
    
    # Generate random diagonal matrix D with ±1 entries
    D = np.diag(np.random.choice([-1, 1], size=n_power))
    
    # Compute randomized Hadamard transform
    Omega_full = (1/np.sqrt(n_power)) * D @ H
    
    # If n < n_power, truncate the matrix to size n
    if n < n_power:
        Omega_full = Omega_full[:n, :]
    
    # Randomly subsample d columns
    subsample_indices = np.random.choice(n_power, size=d, replace=False)
    Omega_L = Omega_full[:, subsample_indices]

    # Step 2: Compute B = (A A^T)^w A Omega_L
    B = A @ Omega_L  # Initial multiplication: A Omega_L
    for _ in range(w):
        B = (A @ A.T) @ B  # Power iteration: (A A^T) B

    # Step 3: Compute the orthogonal basis Q using QR decomposition
    Q, _ = np.linalg.qr(B)

    return Q

def fjlt_compression(A, r, rOV, w):
    m, n = A.shape
    d = r + rOV  # Effective reduced dimension

    # Find smallest power of 2 >= n
    n_power = 2**int(np.ceil(np.log2(n)))

    # P: Random diagonal matrix with ±1 entries
    P = np.random.choice([-1, 1], size=n_power)
    
    # D: Sparse random projection matrix
    q = 0.1
    D = (1/np.sqrt(q)) * np.random.choice([0, 1, -1], size=(n_power, d), 
                                        p=[1-q, q/2, q/2])
    
    # H: Fast Walsh-Hadamard Transform via FFT
    def apply_hadamard(x):
        n_pow = len(x)
        # Pad x to n_power if necessary
        if len(x) < n_pow:
            x = np.pad(x, (0, n_pow - len(x)), mode='constant')
        return fft(x) * (1/np.sqrt(n_pow))
    
    # Construct Omega_L = D.T @ H @ P (applied to columns of A)
    # Step 1: Apply P (element-wise multiplication) to each column of A
    if n < n_power:
        A_padded = np.pad(A, ((0, 0), (0, n_power - n)), mode='constant')
    else:
        A_padded = A
    
    PA = A_padded * P[np.newaxis, :]  # (m × n_power)
    
    # Step 2: Apply Hadamard transform to each row of PA
    HPA = np.apply_along_axis(apply_hadamard, 1, PA)  # (m × n_power)
    
    # Step 3: Apply sparse projection D.T
    Omega_L = HPA @ D  # (m × n_power) @ (n_power × d) = (m × d)
    
    # Step 2: Compute B = (A A^T)^w A Omega_L
    B = Omega_L  # Since Omega_L is already m × d, no need for A @ Omega_L here
    for _ in range(w):
        B = (A @ A.T) @ B
    
    # Step 3: QR decomposition
    Q, _ = np.linalg.qr(B)
    return Q

def cwt_compression(A, r, rOV, w):
    m, n = A.shape
    d = r + rOV  # Effective reduced dimension

    # Step 1: Construct Clarkson-Woodruff Transform (CountSketch matrix)
    # Parameters for sparsity (s controls the number of non-zero entries per column)
    s = max(1, int(np.ceil(np.log(n) / np.log(d))))  # Adjust sparsity based on problem size
    
    # Hash functions for column indices and signs
    hash_indices = np.random.randint(0, d, size=n)  # h: [n] -> [d]
    signs = np.random.choice([-1, 1], size=n)       # σ: [n] -> {-1, 1}
    
    # Build sparse sketching matrix S (d × n)
    S = np.zeros((d, n))
    for j in range(n):
        S[hash_indices[j], j] = signs[j]
    
    # Step 2: Compute sketch B = A S^T
    B = A @ S.T  # (m × n) @ (n × d) = (m × d)
    
    # Step 3: Power iteration: B = (A A^T)^w B
    for _ in range(w):
        B = (A @ A.T) @ B
    
    # Step 4: QR decomposition to get orthonormal basis
    Q, _ = np.linalg.qr(B)
    
    return Q

def randomized_nmf(A, r, r_ov=5, w=1, max_iter=500, tol=1e-6,compression = 'standard'):
    """
    NMF with compression matrices L and R, random initialization for Y_k,
    and multiplicative updates with proper dimension handling.
    """
    m, n = A.shape
    d = r + r_ov  # Effective reduced dimension

    # --- Step 1: Compute compression matrices L and R ---
    if compression == 'standard':
        L = standard_compression(A, r, r_ov, w)      # m × d
        R = standard_compression(A.T, r, r_ov, w).T  # d × n
    elif compression == 'srht':
        L = srht_compression(A, r, r_ov, w)      # m × d
        R = srht_compression(A.T, r, r_ov, w).T  # d × n
    elif compression == 'fjlt':
        L = fjlt_compression(A, r, r_ov, w)      # m × d
        R = fjlt_compression(A.T, r, r_ov, w).T  # d × n
    elif compression == 'cwt':
        L = cwt_compression(A, r, r_ov, w)      # m × d
        R = cwt_compression(A.T, r, r_ov, w).T  # d × n
    else:
        print('Please select compression Algorithm')
        return

    # --- Step 3: Initialize Y_k as random non-negative ---
    Y_k = np.abs(np.random.randn(r, n))  # r × n
    Y_k = np.maximum(Y_k, 1e-6)          # Enforce non-negativity

    # Initialize X
    X_k = np.abs(np.random.randn(m, r))   # m × r
    X_k = np.maximum(X_k, 1e-6)

    # --- Main loop ---
    errors = []
    for k in range(max_iter):
        # --- Update X_{k+1} ---
        # numerator: (m × d) @ (d × r) = m × r
        numerator = (A @ R.T) @ (R @ Y_k.T)
        # denominator: (m × r) @ (r × d) @ (d × r) = m × r
        denominator = X_k @ (Y_k @ R.T @ R @ Y_k.T)
        X_k_plus1 = X_k * numerator / (denominator + 1e-10)
        X_k_plus1 = np.maximum(X_k_plus1, 1e-6)

        # --- Update Y_{k+1} ---
        # numerator: (r × d) @ (d × n) = r × n
        numerator = (L.T @ X_k_plus1).T @ (L.T @ A)
        # denominator: (r × d) @ (d × r) @ (r × n) = r × n
        denominator = (L.T @ X_k_plus1).T @ (L.T @ X_k_plus1) @ Y_k
        Y_k_plus1 = Y_k * numerator / (denominator + 1e-10)
        Y_k_plus1 = np.maximum(Y_k_plus1, 1e-6)

        # --- Convergence check ---
        error = np.linalg.norm(A - X_k_plus1 @ Y_k_plus1, 'fro') / np.linalg.norm(A, 'fro')
        errors.append(error)

        X_k, Y_k = X_k_plus1, Y_k_plus1

    return X_k, Y_k, errors