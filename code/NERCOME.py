import numpy as np

def get_nercome_covariance(x, s, ndraws=500, verbose = True, cov = None):
    ''' Computes the covariance matrix using the
        Non-parametric Eigenvalue-Regularized COvariance Matrix Estimator
        (NERCOME) described in Joachimi 2017
        
        From https://github.com/julianbautista/nercome/blob/main/python/nercome.py
  
        Input
        -----
        x : np.array with shape (n_realisations, n_variables)
        s : index (between 0 and n_realisations-1) where to divide the realisations
        ndraws: number of random realisations of the partitions

        Returns
        -----
        cov_nercome : the NERCOME covariance matrix
    '''
    nreal, nbins = x.shape
    assert s < nreal
    idx = np.arange(nreal)
    
    if verbose:
        sample_cov = x.T @ x/nreal
    
    #-- Compute the matrix many times and then average it
    cov_nercome = np.zeros((nbins, nbins))
    for i in range(ndraws):
        #-- Randomly divide all realisations in two batches
        choice = np.random.choice(idx, size=s, replace=False)
        selection_1 = np.in1d(idx, choice)
        selection_2 = ~selection_1
        x1 = x[selection_1]
        x2 = x[selection_2]
        #-- Estimate sample covariance matrices of the two samples
        cov_sample_1 = np.cov(x1.T)
        cov_sample_2 = np.cov(x2.T)
        #-- Extract eigen values and vectors (we only use vectors)
        #-- which make the U_1 matrix in Eq. 2
        eigvals_1, eigvects_1 = np.linalg.eigh(cov_sample_1)
        #-- Evaluating Eq. 2:
        #-- first U_1^T S_2 U_1
        mid_matrix = eigvects_1.T @ cov_sample_2 @ eigvects_1
        #-- make it a diagonal matrix
        mid_matrix_diag = np.diag(np.diag(mid_matrix))
        #-- now compute Z = U_1 diag( U_1^T S_2 U_1 ) U_1^T
        cov_nercome += eigvects_1 @ mid_matrix_diag @ eigvects_1.T
        
        if verbose:
            if i % (ndraws // 20) == 0:
                print("PRIAL epoch", i, ":", 1 - np.linalg.norm(cov_nercome/(i+1) - cov)**2/np.linalg.norm(sample_cov - cov)**2)
    cov_nercome /= ndraws
    return cov_nercome

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    from sklearn.covariance import LedoitWolf
    from QuEST import QuEST, NL_covariance
    
    n = 2000
    p = 200
    s = n//2
    ndraws = 1000
    tau_pop = np.concatenate([np.linspace(1.9, 2.1, p//2), np.linspace(4.9, 5.1, p - p//2)])
        
    # Sampling
    cov = np.diag(tau_pop)
    X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))
        
    # Estimation
    start_time = time.time()
    print("NERCOME, ndraws =", ndraws, ", n =", n, ", p =", p)
    nercome_covariance = get_nercome_covariance(X_sample, s, ndraws=ndraws, verbose = False, cov = cov)
    last_time = time.time()
    print("Compute time:", last_time - start_time, "s.")
    
    # Shrinkage
    sample_cov = X_sample.T @ X_sample/n
    tau_cov, _ = np.linalg.eigh(sample_cov)
    LW_cov = LedoitWolf().fit(X_sample).covariance_
    ex_shrunk_covariance = NL_covariance(tau_pop, X_sample, assume_centered = True)
    print("Lin covariance PRIAL:\t", 1 - np.linalg.norm(LW_cov - cov)**2/np.linalg.norm(sample_cov - cov)**2)
    print("NERCOME covariance PRIAL:\t", 1 - np.linalg.norm(nercome_covariance - cov)**2/np.linalg.norm(sample_cov - cov)**2)
    print("Ex. NL covariance PRIAL:", 1 - np.linalg.norm(ex_shrunk_covariance - cov)**2/np.linalg.norm(sample_cov - cov)**2)
    
    # Initial plot
    lambda_unsrt1, dlambdadtau_unsrt1, x1, f1, x_tilde1, F1 =  QuEST(tau_pop, p, n)
    lambda_unsrt3, dlambdadtau_unsrt3, x3, f3, x_tilde3, F3 =  QuEST(tau_cov, p, n)
    plt.figure()
    plt.plot(x1[0], f1[0], label=r'Sample PDF from pop')
    # plt.plot(x3[0], f3[0], label=r'Sample PDF from sample spectrum')
    plt.title("Sample PDF")
    plt.legend()
    plt.show()