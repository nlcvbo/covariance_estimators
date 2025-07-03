import numpy as np

def H_Ty(X, sigma):
    n,p = X.shape
    weights = np.diag(1/np.diag(X @ np.linalg.pinv(sigma) @ X.T))
    return p/n*(X.T @ weights @ X)

def SRTy_estimator(X, alpha = 0.1, assume_centered = False, eps = 1e-6, max_iter = 100000):
    """
    https://inria.hal.science/hal-02485823/document
    """    
    n,p = X.shape
    target = np.eye(p)
    if not assume_centered:
        X -= X.mean(axis=0)[np.newaxis,:]
    sigma = np.eye(p)
    sigma_old = np.eye(p)
    criterion = True
    count_iter = 0
    while criterion:
        sigma_old = sigma
        D, U = np.linalg.eigh(H_Ty(X, sigma_old))
        sigma = U @ np.diag(D/(1+alpha)+alpha/(1+alpha)*np.diag(target)) @ U.T
        criterion = (np.linalg.norm(sigma - sigma_old, ord='fro')**2/p > eps*np.linalg.norm(sigma_old, ord='fro')*np.linalg.norm(sigma, ord='fro')/p) or (count_iter > max_iter)
    return sigma

def nu_estimator(X, scatter, assume_centered = False, method = "Hill", nu_max = 12):
    """
        nu estimation
        https://github.com/convexfi/fitHeavyTail/blob/master/R/nu_OPP_estimator.R
    """
    n,p = X.shape
    if not assume_centered:
        X -= X.mean(axis=0)[np.newaxis,:]
        n -= 1
    sample = X.T @ X/n
    var_X = np.trace(sample)
    if method == "OPP":
        eta = var_X/np.trace(scatter)
    elif method == "OPP_harmonic":
        r2 = np.diag(X @ np.linalg.pinv(scatter) @ X.T)
        eta = var_X/np.trace(scatter)*(p/r2).sum()
    elif method == "POP":
        """
        https://centralesupelec.hal.science/hal-03436848/document
        """
        nu = nu_estimator(X, scatter, assume_centered = assume_centered, method = "OPP_harmonic")
        r2 = np.diag(X @ np.linalg.pinv(scatter) @ X.T)
        u = (p + nu)/(nu + r2*n/(n-1))
        r2i = r2/(1 - r2*u/n)
        eta = (1 - p/n) * r2i.sum()/n/p
    elif method == "Hill":
        """
        [AshurbekovaCarleveForbesAchard2020] eq (28)
        https://www.researchgate.net/publication/337495975_Optimal_shrinkage_for_robust_covariance_matrix_estimators_in_a_small_sample_size_setting
        """
        b = 4/(nu_max + 4)
        kn = int(n**b)
        norm_X = np.sqrt((X**2).sum(axis=1))
        sorted_norm_X = np.sort(norm_X)[::-1]
        nu = 1/(np.log(sorted_norm_X[:kn]/sorted_norm_X[kn]).mean())
        eta = nu/(nu-2)
    return 2*eta/(eta-1)

def Ty_estimator(X, rho = 0.1, assume_centered = False, eps = 1e-6, max_iter = 100000):
    """
    Tyler M-estimator
    https://arxiv.org/pdf/1401.6926.pdf
    """    
    n,p = X.shape
    if not assume_centered:
        X -= X.mean(axis=0)[np.newaxis,:]
    X /= (X**2).sum(axis=1)[:,np.newaxis]/p # ?
    sigma = np.eye(p)
    sigma_old = np.eye(p)
    criterion = True
    count_iter = 0
    while criterion:
        sigma_old = sigma
        weights = np.diag(1/np.diag(X @ np.linalg.pinv(sigma_old) @ X.T))
        sigma = (1-rho)*p/n*(X.T @ weights @ X) + rho*np.eye(p)
        sigma = p*sigma/np.trace(sigma)
        criterion = (np.linalg.norm(sigma - sigma_old, ord='fro')**2/p > eps*np.linalg.norm(sigma_old, ord='fro')*np.linalg.norm(sigma, ord='fro')/p) or (count_iter > max_iter)
    return sigma
    
def cov_SRTy_estimator(X_sample, alpha = 0.1, assume_centered = False, eps = 1e-6, max_iter = 10000, method = "OPP"):
    X = X_sample.copy()
    scatter = SRTy_estimator(X, alpha, assume_centered, eps, max_iter)
    nu = nu_estimator(X, scatter, assume_centered, method = method)
    return nu/(nu-2)*scatter
    
def cov_Ty_estimator(X_sample, rho = 0.1, assume_centered = False, eps = 1e-6, max_iter = 10000, method = "OPP"):
    X = X_sample.copy()
    scatter = Ty_estimator(X, rho, assume_centered, eps, max_iter)
    nu = nu_estimator(X, scatter, assume_centered, method = method)
    return nu/(nu-2)*scatter

def cov_Ashurbekova_estimator(X_sample, assume_centered = False, distrib="Gaussian", nu_max=12, eps = 1e-6, max_iter = 10000):
    """
    https://www.researchgate.net/publication/337495975_Optimal_shrinkage_for_robust_covariance_matrix_estimators_in_a_small_sample_size_setting
    """
    X = X_sample.copy()
    n,p = X.shape
    mu = X.mean(axis=0)[np.newaxis,:]
    weights = 1/((X-mu)**2).sum(axis=1)
    S = (X-mu).T @ np.diag(weights) @ (X-mu)/n
    zeta = p*np.trace(S @ S) - p/n - 1
    if distrib == "Gaussian":
        beta = n*zeta/((n+1)*zeta+p+1)
        alpha = 1-beta
        sigma = beta*(X-mu).T @ (X-mu)/n + alpha*np.eye(p)
    elif distrib == "Student":     
        nu = nu_estimator(X, None, assume_centered = False, method = "Hill", nu_max = nu_max)
        beta = n*zeta/((n-1+2*(nu+p)/(nu+p+2))*zeta + (nu+p)/(nu+p+2)*(p+2) - 1)
        alpha = 1-beta
        def u(d):
            return (p+nu)/(d+nu)   
        sigma = np.eye(p)
        sigma_old = np.eye(p)
        criterion = True
        count_iter = 0
        while criterion:
            sigma_old = sigma
            weights = np.diag(u((X-mu) @ np.linalg.pinv(sigma) @ (X-mu).T))
            sigma = beta*(X-mu).T @ (X-mu)/n + alpha*np.eye(p)
            criterion = (np.linalg.norm(sigma - sigma_old, ord='fro')**2/p > eps*np.linalg.norm(sigma_old, ord='fro')*np.linalg.norm(sigma, ord='fro')/p) or (count_iter > max_iter)
    return sigma*np.trace((X-mu).T @ (X-mu)/(n-1))/np.trace(sigma)

def cov_Ashurbekova2_estimator(X_sample, assume_centered = False, nu_max=12, eps = 1e-6, max_iter = 10000):
    """
    https://www.researchgate.net/publication/337495975_Optimal_shrinkage_for_robust_covariance_matrix_estimators_in_a_small_sample_size_setting
    """
    X = X_sample.copy()
    n,p = X.shape
    
    mu = X.mean(axis=0)[np.newaxis,:]
    weights = 1/((X-mu)**2).sum(axis=1)
    S = (X-mu).T @ np.diag(weights) @ (X-mu)/n
    zeta = p*np.trace(S @ S) - p/n - 1
    nu = nu_estimator(X, None, assume_centered = False, method = "Hill", nu_max = nu_max)
    beta = n*zeta/((n-1+2*(nu+p)/(nu+p+2))*zeta + (nu+p)/(nu+p+2)*(p+2) - 1)
    alpha = 1-beta
    
    V_old = np.eye(p)
    mu_old = X.mean(axis=0)[np.newaxis,:]
    V = np.eye(p)
    mu = X.mean(axis=0)[np.newaxis,:]
    
    criterion = True
    count_iter = 0
    while criterion:
        V_old = V
        mu_old = mu
        weights = np.diag((nu + p)/(nu + (X - mu_old) @ np.linalg.pinv(V_old) @ (X - mu_old).T))
        mu = (np.diag(weights) @ X).sum(axis=0)[np.newaxis,:]/weights.sum()
        weights = np.diag(1/(nu + (X - mu_old) @ np.linalg.pinv(V_old) @ (X - mu_old).T))
        V = beta*(p+nu)/n*(X.T @ np.diag(weights) @ X) + alpha*np.eye(p)
        V = p*V/np.trace(V)
        criterion = (np.linalg.norm(V - V_old, ord='fro')**2/p > eps*np.linalg.norm(V_old, ord='fro')*np.linalg.norm(V, ord='fro')/p) or (count_iter > max_iter)
    return V*np.trace((X-mu).T @ (X-mu)/(n-1))/np.trace(V)