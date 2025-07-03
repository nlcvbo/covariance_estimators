import numpy as np

def LWO_estimator(X, assume_centered = False, S_r = None):
    """    
    Parameters
    ----------
    X : np.array of shape (n,p)
        Array of n iid samples of dimension p
    assume_centered : boolean, optional
        If the true mean of the distribution is known to be 0, set assume_centered = True. Otherwise, the empirical mean will be removed. The default is False.
        Note that if you manually removed the empirical mean previously, you should set assume_centered = False.
    S_r : None or np.array of shape (p,p), optional
        The reference matrix to shrink with, np.array of shape (p,p). If set to known, the shrinkage will be done with S_r = np.eye(p) . The default is None.

    Returns
    -------
    S_star : np.array of shape (p,p)
        The estimated covariance using linear shrinkage.

    """
    X = X.T
    if not assume_centered:
        S_star = LWO_estimator_unknown_mean(X, S_r)
    else:
        S_star = LWO_estimator_known_mean(X, S_r)
    return S_star

def LWO_estimator_known_mean(X, S_r = None):
    """
    Parameters
    ----------
    X : np.array of shape (p,n)
        Data array of n iid samples of dimension p drawn from a distribution of known mean 0.
    S_r : None or np.array of shape (p,p)
        The reference matrix to shrink with, np.array of shape (p,p). If set to known, the shrinkage will be done with S_r = np.eye(p). The default is None.

    Returns
    -------
    S_star : np.array of shape (p,p)
        The estimated covariance using linear shrinkage.
    """
    p, n = X.shape
    Id = np.eye(p)
    S = X @ X.T/n
    
    
    try:
        if S_r == None:
            S_r = Id
    except ValueError: 
        if (S_r**2).sum() == 0:
            S_r = Id
    S_r /= np.linalg.norm(S_r, ord = 'fro')/np.sqrt(p)
    
    mu = np.trace(S @ S_r.T)/p  
    delta2 = np.linalg.norm(S - mu*S_r, ord = 'fro')**2/p
    beta2 = (((X**2).sum(axis=0)**2).sum()/n - np.linalg.norm(S, ord = 'fro')**2)/p/n
    beta2 = n/(n-1)*beta2
    beta2 = min(beta2, delta2)
    
    shrinkage = beta2/delta2  
    return shrinkage*mu*S_r + (1 - shrinkage)*S

def LWO_estimator_unknown_mean(X, S_r = None):
    """
    Parameters
    ----------
    X : np.array of shape (p,n)
        Data array of n iid samples of dimension p from a distribution of unknown mean.
    S_r : None or np.array of shape (p,p)
        The reference matrix to shrink with, np.array of shape (p,p). If set to known, the shrinkage will be done with S_r = np.eye(p). The default is None.

    Returns
    -------
    S_star : np.array of shape (p,p)
        The estimated covariance using linear shrinkage.
    """
    p, n = X.shape
    Id = np.eye(p)
    X = X - X.mean(axis=1)[:,np.newaxis]
    S = X @ X.T/(n-1)
    
    try:
        if S_r == None:
            S_r = Id
    except ValueError: 
        if (S_r**2).sum() == 0:
            S_r = Id
    S_r /= np.linalg.norm(S_r, ord = 'fro')/np.sqrt(p)
    
    mu_I = np.trace(S)/p
    mu = np.trace(S @ S_r.T)/p  
    delta2 = np.linalg.norm(S - mu*S_r, ord = 'fro')**2/p
    
    gamma, lambd = n*(n-1)/(n**2-3*n+3), (n-2)*n**2/(n**2-3*n+3)/(n-1)
    q0, q1, q2 = (n-2)/p/(n-1), 1/p/(n-1), (p-1)/p/(n-1)
    q = np.array([q0, q1, -q2])
    c0, c1, c2 = 1/gamma - 1/n - lambd/gamma/n**2, lambd/gamma/n**2, lambd/gamma*(p+1)/n**2
    c = np.array([c0, c1, c2])
    c_f = c + (c1-c2)*q/(1-q1-q2)
    
    beta2_bar = ((X**2).sum(axis=0)**2).sum()/p/(n-1)**2 - np.linalg.norm(S, ord = 'fro')**2/p/n
    beta2 = (beta2_bar - c_f[1]*delta2 - c_f[2]*mu_I**2 - c_f[1]*(mu**2 - mu_I**2))/c_f[0]
    beta2 = max(beta2,0)
    beta2 = min(beta2, delta2)
    
    shrinkage = beta2/delta2
    return shrinkage*mu*S_r + (1 - shrinkage)*S
