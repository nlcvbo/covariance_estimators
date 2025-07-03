import numpy as np
import scipy as sp
from scipy.optimize import minimize, fsolve
import matplotlib.pyplot as plt
import warnings

# Support identification

def phi_prime(u, w, t, c):
    """
        Inputs:
            u: float array of shape (L)
            w: weights, array of shape (K)
            t: dirac points of eigenvalue distrib, array of shape (K)
            c: float, p/n
        Outputs:
            phi_p: float array of shape (L), phi'(u)
    """
    # Derivative of phi, in order to find the minima of phi on each non-discarded interval
    phi_p = 2*(w[np.newaxis,:]*t[np.newaxis,:]**2/(t[np.newaxis,:]-u[:,np.newaxis])**3).sum(axis=1)
    return phi_p

def phi_prime2(u, w, t, c):
    """
        Inputs:
            u: float array of shape (L)
            w: weights, array of shape (K)
            t: dirac points of eigenvalue distrib, array of shape (K)
            c: float, p/n
        Outputs:
            phi_p2: float, sum of phi'(u)^2
    """
    # Derivative of phi, in order to find the minima of phi on each non-discarded interval
    phi_p = phi_prime(u, w, t, c)
    return (phi_p**2).sum()

def phi_tilde(u, w, t, c):
    """
        Inputs:
            u: float array of shape (L)
            w: weights, array of shape (K)
            t: dirac points of eigenvalue distrib, array of shape (K)
            c: float, p/n
        Outputs:
            phi_m: float array of shape (L), phi(u) - 1/c
    """
    # phi - 1/c, equals 0 where the support ends
    phi = (w[np.newaxis,:]*t[np.newaxis,:]**2/(t[np.newaxis,:]-u[:,np.newaxis])**2).sum(axis=1)
    phi_t = phi - 1/c
    return phi_t

def phi_tilde2(u, w, t, c):
    """
        Inputs:
            u: float array of shape (L)
            w: weights, array of shape (K)
            t: dirac points of eigenvalue distrib, array of shape (K)
            c: float, p/n
        Outputs:
            phi_m2: float, sum of (phi(u) - 1/c)^2
    """
    phi_t = phi_tilde(u, w, t, c)
    return (phi_t**2).sum()

def necessary_condition(w, t, c):
    """
        Inputs:
            w: weights, array of shape (K)
            t: dirac positive points of eigenvalue distrib, array of shape (K)
            c: float, p/n
        Outputs:
            nonseparation: bool array of shape (K-1), nonseparation[k] == True if there is no spectral separation in [t_k, t_k+1] (necessary condition for spectral separation not met)
    """
    # Compute lower bound of phi on each interval [t_k, t_k+1]
    K = w.shape[0]
    eps = 1e-14
    
    hat_xk = (t[:-1]*t[1:])**(2/3)*(w[:-1]**(1/3)*t[1:]**(1/3) + t[:-1]**(1/3)*w[1:]**(1/3))/(w[:-1]**(1/3)*t[:-1]**(2/3) + t[1:]**(2/3)*w[1:]**(1/3))
    
    # Handle the case when there is only one interval and when there is more
    if K > 1:
        # The sum is useful only for subdiagonal terms, to avoid Divide warnings, we fix the denominator
        den = (t[np.newaxis,:-2] - t[2:,np.newaxis])**2
        den[den == 0] = 1
        
        cum_tab1 = (w[np.newaxis,:-2]*t[np.newaxis,:-2]**2/den).cumsum(axis=1)
        cum_tab1 = np.concatenate([np.zeros(1), np.diag(cum_tab1)], axis=0)
        
        cum_tab2 = (w[np.newaxis,2:]*t[np.newaxis,2:]**2/den)[:,::-1].cumsum(axis=1)[::-1]
        cum_tab2 = np.concatenate([np.diag(cum_tab2), np.zeros(1)], axis=0)
    else:
        cum_tab1 = np.zeros(0)
        cum_tab2 = np.zeros(0)
    
    lower_bound = w[:-1]*t[:-1]**2/(t[:-1] - hat_xk + eps)**2 + w[1:]*t[1:]**2/(t[1:] - hat_xk + eps)**2 + cum_tab1 + cum_tab2
    
    # Index bool tab of length K-1, nonseparation[k] == True if there is no spectral separation in [t_k, t_k+1] (necessary condition for spectral separation not met)
    nonseparation = lower_bound > 1/c
    
    # nonseparation bool array of shape (K-1)
    return nonseparation.astype(bool)

def necessary_sufficient_condition(w, t, c):
    """
        Inputs:
            w: weights, array of shape (K)
            t: dirac positive points of eigenvalue distrib, array of shape (K)
            c: float, p/n
        Outputs:
            separation: bool array of shape (K-1), separation[k] == True iif there is spectral separation in [t_k, t_k+1]
            x_star: float array of shape (K-1), x_star[k] is the argmin of phi_tilde on each interval [t_k, t_k+1] where separation[k] == True, x_star[k] == 0 otherwise
    """
    K = w.shape[0]
    
    # Retrieving some intervals where there is no spectral separation for sure
    nonseparation = necessary_condition(w, t, c)
    
    # Return if there is no interval left for separation
    if (~nonseparation).sum() == 0:
        separation = np.zeros(K-1, dtype=bool)
        x_star = np.zeros(K-1)
        return separation, x_star
    
    # Pre-computing hat_xk
    hat_xk = (t[:-1]*t[1:])**(2/3)*(w[:-1]**(1/3)*t[1:]**(1/3) + t[:-1]**(1/3)*w[1:]**(1/3))/(w[:-1]**(1/3)*t[:-1]**(2/3) + t[1:]**(2/3)*w[1:]**(1/3))
    
    # Indexes of k and k+1 for intervals not discarded by necessary condition
    idx = np.concatenate([(~nonseparation), np.zeros(1, dtype=bool)], dtype=bool) # indexes of k
    idxp = np.concatenate([np.zeros(1, dtype=bool), (~nonseparation)], dtype=bool) # indexes of k+1
    
    # t_k, w_k, hat_xk, phip_xk
    tp = t[idx]
    wp = w[idx]
    hat_xkp = hat_xk[(~nonseparation)]
    phip_xkp = phi_prime(hat_xkp, w, t, c)
    
    # t_k+1, w_k+1
    tpp = t[idxp]
    wpp = w[idxp]
    
    # Computing (vectorized) lower and upper bounds [x_down, x_up] for root solving
    x_up = hat_xkp
    x_down = hat_xkp
    
    idx_up = phip_xkp > 0
    idx_down = phip_xkp < 0
    
    x_up[idx_down] = tpp[idx_down] - (2*wpp[idx_down]*tpp[idx_down]**2/(-2*wp[idx_down]*tp[idx_down]**2/(tp[idx_down] - hat_xkp[idx_down])**3 - phip_xkp[idx_down]))**(1/3)    
    x_down[idx_up] = tp[idx_up] + (2*wp[idx_up]*tp[idx_up]**2/(2*wpp[idx_up]*tpp[idx_up]**2/(tpp[idx_up] - hat_xkp[idx_up])**3 + phip_xkp[idx_up]))**(1/3)
    
    # Find the root x_star of phi_prime on those intervals
    x_star = np.zeros(K-1)
    x_star[(~nonseparation)] = minimize(phi_prime2, x0 = (x_down + x_up)/2, args = (w, t, c), bounds = [(x_down[i], x_up[i]) for i in range(x_down.shape[0])]).x
    
    # Index bool tab of length K-1, separation[k] == True iif there is spectral separation in [t_k, t_k+1]
    separation = (~nonseparation)
    separation[(~nonseparation)] = phi_prime(x_star[(~nonseparation)], w, t, c) < 1/c
    
    # separation bool array of shape (K-1), x_star float array of shape (K-1)
    return separation.astype(bool), x_star


def interval_boundaries(w, t, c):
    """
        Inputs:
            w: weights, array of shape (K)
            t: dirac positive points of eigenvalue distrib, array of shape (K)
            c: float, p/n
        Outputs:
            separation: bool array of shape (K-1), separation[k] == True iif there is spectral separation in [t_k, t_k+1]
            x_min_left: float array of shape (separation.sum()), x_min_left[separation[:k+1].sum()] is the support endpoint on [x_star_k, t_k+1] for k s.t separation[k] == True
            x_min_right: float array of shape (separation.sum()), x_min_right[separation[:k+1].sum()] is the support endpoint on [x_star_k, t_k+1] for k s.t separation[k] == True
    """
    # Retrieving interval indexes where there is spectral separation, and x_star the argmin of phi on those intervals
    separation, x_star = necessary_sufficient_condition(w, t, c)
    
    # Return if there is no interval left for separation
    if separation.sum() == 0:
        x_min_left = np.zeros(0)
        x_min_right = np.zeros(0)
        return separation, x_min_left, x_min_right
    
    # Indexes of k and k+1 for intervals where there is spectral separation
    idx = np.concatenate([separation, np.zeros(1, dtype=bool)], dtype=bool)
    idxp = np.concatenate([np.zeros(1, dtype=bool), separation], dtype=bool)
    
    # t_k, w_k, x_stark
    tp = t[idx]
    wp = w[idx]
    x_starp = x_star[separation]
    
    # t_k+1, w_k+1
    tpp = t[idxp]
    wpp = w[idxp]
    
    
    # Support endpoint on [t_k, x_star_k], x_min_left is found in the sub-interval [x_down, x_up] by root-finding of phi_tilde
    num = wp*tp**2
    tab1 = (w[np.newaxis,:]*t[np.newaxis,:]**2/(t[np.newaxis,:]-x_star[:,np.newaxis])**2)[:,::-1].cumsum(axis=1)[:,::-1][:,idxp]
    tab2 = (w[np.newaxis,:]*t[np.newaxis,:]**2/(t[np.newaxis,:]-t[:-1,np.newaxis] + 1.*(t[np.newaxis,:] == t[:-1,np.newaxis]))**2)[:,::-1].cumsum(axis=1)[:,::-1][:,idxp]
    den = wp*tp**2/(tp-x_starp)**2 - phi_tilde(x_starp, w, t, c) + np.diag(tab1) - np.diag(tab2)
    # Bounds [x_down, x_up]
    x_down, x_up = np.zeros(separation.sum()), np.zeros(separation.sum())
    x_down = x_starp 
    x_down[den > 0] += tp[den > 0] - x_starp[den > 0] + np.sqrt(num[den > 0]/den[den > 0])
    x_up = x_starp
    # Root finding
    x_min_left = minimize(phi_tilde2, x0 = (x_down + x_up)/2, args = (w, t, c), bounds = [(x_down[i], x_up[i]) for i in range(x_down.shape[0])]).x
    
    # Support endpoint on [x_star_k, t_k+1], x_min_right is found in the sub-interval [x_down, x_up] by root-finding of phi_tilde
    num = wpp*tpp**2
    tab1 = np.concatenate([np.zeros((x_star.shape[0],1)), (w[np.newaxis,:]*t[np.newaxis,:]**2/(t[np.newaxis,:]-x_star[:,np.newaxis])**2).cumsum(axis=1)[:,:-1]], axis=1)[:,idx]
    tab2 = np.concatenate([np.zeros((x_star.shape[0],1)), (w[np.newaxis,:]*t[np.newaxis,:]**2/(t[np.newaxis,:]-t[1:,np.newaxis] + 1.*(t[np.newaxis,:] == t[1:,np.newaxis]))**2).cumsum(axis=1)[:,:-1]], axis=1)[:,idx]
    den = wpp*tpp**2/(tpp-x_starp)**2 - phi_tilde(x_starp, w, t, c) + np.diag(tab1) - np.diag(tab2)
    # Bounds [x_down, x_up]
    x_down, x_up = np.zeros(separation.sum()), np.zeros(separation.sum())
    x_down = x_starp
    x_up = x_starp
    x_up[den > 0] += tpp[den > 0] - x_starp[den > 0] - np.sqrt(num[den > 0]/den[den > 0])
    # Root finding
    x_min_right = minimize(phi_tilde2, x0 = (x_down + x_up)/2, args = (w, t, c), bounds = [(x_down[i], x_up[i]) for i in range(x_down.shape[0])]).x
    
    # separation bool array of shape (K-1), x_min_left and x_min_right of shape (separation.sum())
    return separation, x_min_left, x_min_right

def minimum_support(w, t, c):
    """
        Inputs:
            w: weights, array of shape (K)
            t: dirac positive points of eigenvalue distrib, array of shape (K)
            c: float, p/n
        Outputs:
            min_support: float array of shape (1), support endpoint on [-np.inf, t_0]
    """
    # Support endpoint on [-np.inf, t_0], min_support is found in the sub-interval [x_down, x_up] by root-finding of phi_tilde
    x_down = (t[0] - np.sqrt(c*(w*t**2).sum()) - 1)*np.ones(1)
    x_up = (t[0] - np.sqrt(c*w[0]*t[0]**2)/2)*np.ones(1)
    # Root finding
    min_support = minimize(phi_tilde2, x0 = (x_down + x_up)/2, args = (w, t, c), bounds = [(x_down[i], x_up[i]) for i in range(x_down.shape[0])]).x
    
    # min_support of shape (1)
    return min_support

def maximum_support(w, t, c):
    """
        Inputs:
            w: weights, array of shape (K)
            t: dirac positive points of eigenvalue distrib, array of shape (K)
            c: float, p/n
        Outputs:
            max_support: float array of shape (1), support endpoint on [t_-1, np.inf]
    """
    # Support endpoint on [t_-1, np.inf], max_support is found in the sub-interval [x_down, x_up] by root-finding of phi_tilde
    x_down = (t[-1] + np.sqrt(c*w[-1]*t[-1]**2)/2)*np.ones(1)
    x_up = (t[-1] + np.sqrt(c*(w*t**2).sum()) + 1)*np.ones(1)
    # Root finding
    max_support = minimize(phi_tilde2, x0 = (x_down + x_up)/2, args = (w, t, c), bounds = [(x_down[i], x_up[i]) for i in range(x_down.shape[0])]).x
    
    # max_support of shape (1)
    return max_support

def support(tau, c):
    """
        Inputs:
            tau: array of shape (p), population eigenvalues
            c: float, p/n
        Outputs:
            nu: int, number of distinct intervals that constitute the support
            u_nu: float array of shape (2*nu), the support endpoints in increasing order
            omega_nu: float array of shape (nu), omega_nu[k] contains how many population eigenvalues correspond to the support interval [u_nu[2*k], u_nu[2*k+1]]
            w: weights, array of shape (K)
            t: dirac positive points of eigenvalue distrib, array of shape (K)
    """
    # Extract positive points and weights
    p = tau.shape[0]
    t, w = np.unique(tau, return_counts=True)
    w = w/p
    # Discarding 0 population eigenvalues and including their weight in the first interval, the support is supposed to be in R_+^*
    t_pos = t[t > 0]
    w_pos = w[t > 0]
    if w_pos.shape[0] > 0:
        w_pos[0] += w[t <= 0].sum()
    else:
        warnings.warn(Warning("All population eigenvalues are non-positive."))
        t_pos = np.linspace(1,2,p)
        w_pos = np.ones(p)
    
    # The support endpoints between [t_0,t_K-1], [-inf, t_0] and [t_K-1,inf] respectively
    separation, x_min_left, x_min_right = interval_boundaries(w_pos, t_pos, c)
    min_support = minimum_support(w_pos, t_pos, c)
    max_support = maximum_support(w_pos, t_pos, c)
    
    # The support endpoints in increasing order
    u_nu = np.sort(np.concatenate([min_support, x_min_left, x_min_right, max_support], axis=0))
    
    # Number of distinct intervals that constitute the support
    nu = u_nu.shape[0]//2
    
    # Indexes 2k (left) and 2k+1 (right)
    idx_left = (2*np.ones(nu).cumsum() - 2).astype(int)
    idx_right = (2*np.ones(nu).cumsum() - 1).astype(int)
    
    # Change the extremities of the interval bounds with -inf and +inf
    # Left and right limits have to be taken care of, particularly the case of 0 eigenvalue, see [Ledoit and Wolf, 2016] part 4.3
    # /!\ The article is quite vague on what to do in this case, I assumed regarding [Bai and Silverstein, 1999] Lemma 2.3 that 0 population eigenvalue are to be included in the first positive support interval
    # /!\ Remark: another way of doing it is not to consider them in the support of F with density, but to include them later in the cdf in F(0)
    u_nu_ext = np.concatenate([-np.inf*np.ones(1), u_nu[1:-1], np.inf*np.ones(1)], axis=0) 
    
    # omega_nu[k] contains how many population eigenvalues correspond to the support interval [u_nu[2*k], u_nu[2*k+1]]
    omega_nu = ((t[:,np.newaxis] - u_nu_ext[np.newaxis, idx_left] > 0) & (t[:,np.newaxis] - u_nu_ext[np.newaxis, idx_right] <= 0)).sum(axis=0)
    
    return nu, u_nu, omega_nu, w_pos, t_pos


# Derivatives

def dphi_du(u, tau):
    """
        Inputs:
            u: array of shape (L), the support endpoints
            tau: array of shape (p), population eigenvalues
        Outputs:
            deriv: array of shape (L), where deriv[i] = (\partial phi_tilde/\partial u)(u[i])
    """
    deriv = 2*(tau[np.newaxis,:]**2/(tau[np.newaxis,:] - u[:,np.newaxis])**3).mean(axis=1)
    return deriv

def dphi_dtau(u, tau):
    """
        Inputs:
            u: array of shape (L), the support endpoints
            tau: array of shape (p), population eigenvalues
        Outputs:
            deriv: array of shape (L,p), where deriv[i,j] = (\partial phi_tilde/\partial tau[j])(u[i])
    """
    p = tau.shape[0]
    deriv = -2*tau[np.newaxis,:]*u[:,np.newaxis]/(tau[np.newaxis,:] - u[:,np.newaxis])**3/p
    return deriv

def du_dtau(u, tau):
    """
        Inputs:
            u: array of shape (L), the support endpoints
            tau: array of shape (p), population eigenvalues
        Outputs:
            deriv: array of shape (L,p), where deriv[i,j] = \partial u[i]/\partial tau[j]
    """
    deriv = -dphi_dtau(u, tau)/dphi_du(u, tau)[:,np.newaxis]
    return deriv
    

# Grid

def grid(nu, u_nu, omega_nu):
    """
        Inputs:
            nu: int, number of distinct intervals that constitute the support
            u_nu: float array of shape (2*nu), the support endpoints in increasing order
            omega_nu: float array of shape (nu), omega_nu[i] contains how many population eigenvalues correspond to the support interval [u_nu[2*i], u_nu[2*i+1]]
        Outputs:
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
    """
    xi = []
    for i in range(nu):
        j = np.ones(omega_nu[i]+2).cumsum() - 1
        xi_i = u_nu[2*i] + (u_nu[2*i+1] - u_nu[2*i])*np.sin(np.pi*j/2/(omega_nu[i]+1))**2
        xi += [xi_i]
    return xi

def dxi_dtau(nu, u_nu, omega_nu, tau):
    """
        Inputs:
            nu: int, number of distinct intervals that constitute the support 
            u_nu: array of shape (2*nu), the support endpoints
            omega_nu: float array of shape (nu), omega_nu[i] contains how many population eigenvalues correspond to the support interval [u_nu[2*i], u_nu[2*i+1]]
            tau: array of shape (p), population eigenvalues
        Outputs:
            deriv: list of nu float array of shape (omega_nu[i]+2, p), where deriv[i][j,k] = \partial xi[i][j]/\partial tau[k]
    """
    # Indexes 2k (left) and 2k+1 (right)
    idx_left = (2*np.ones(nu).cumsum() - 2).astype(int)
    idx_right = (2*np.ones(nu).cumsum() - 1).astype(int)
    # Derivatives of u_2k and u_2k+1
    dudtau = du_dtau(u_nu, tau)
    d_left = dudtau[idx_left]
    d_right = dudtau[idx_right]
    
    # Compute de derviatives interval by interval
    deriv = []
    for i in range(nu):
        j = np.ones(omega_nu[i]+2).cumsum() - 1
        deriv_i = (1 - np.sin(np.pi*j/2/(omega_nu[i]+1))**2)[:,np.newaxis]*d_left[i,np.newaxis,:] + (np.sin(np.pi*j/2/(omega_nu[i]+1))**2)[:,np.newaxis]*d_right[i,np.newaxis,:]
        deriv += [deriv_i]
    return deriv


# Solving the Fundamental Equation

def gamma_i(y, i, tau, xi, c):
    """
        Inputs:
            y: float array of shape (omega_nu[i])
            i: int in [0, nu-1], consider the grid on the i_th support interval
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            c: float, p/n
        Outputs:
            rst: float array of shape (omega_nu[i]), (gamma_j^i(y_j))_{j \in [1, omega_nu[i]]}
    """
    rst = (tau[np.newaxis,:]**2/((tau[np.newaxis,:] - xi[i][1:-1,np.newaxis])**2 + y[:,np.newaxis]**2)).mean(axis=1) - 1/c
    return rst

def gamma_i2(y, i, tau, xi, c):
    """
        Inputs:
            y: float array of shape (omega_nu[i])
            i: int in [0, nu-1], consider the grid on the i_th support interval
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            c: float, p/n
        Outputs:
            rst2: float, sum of (gamma_j^i(y_j)^2)_{j \in [1, omega_nu[i]]}
    """
    rst = gamma_i(y, i, tau, xi, c)
    return (rst**2).sum()

def find_y(w, t, tau, nu, xi, c):
    """
        Inputs:
            w: weights, array of shape (K)
            t: dirac positive points of eigenvalue distrib, array of shape (K)
            tau: array of shape (p), population eigenvalues
            nu: int, number of distinct intervals that constitute the support 
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            c: float, p/n
        Outputs:
            y: list of nu float array of shape (omega_nu[i]+2); imaginary part 
    """
    y = []
    for i in range(nu):
        # y_i is found in the sub-interval [y_down, y_up] by root-finding of gamma_i
        delta_ij = np.min((t[np.newaxis,:]-xi[i][1:-1,np.newaxis])**2, axis=1)
        idx_ij = (t[np.newaxis,:]-xi[i][1:-1,np.newaxis])**2 - delta_ij[:,np.newaxis] == 0
        
        y_down = np.sqrt(np.maximum(0, c*(np.tile(w*t**2, (idx_ij.shape[0],1))*idx_ij).sum(axis=1) - delta_ij))/2
        y_up = np.sqrt(np.maximum(0, c*(w*t**2).sum() - delta_ij)) + 1
        
        # Root finding
        if y_down.shape[0] > 0:
            y_i = minimize(gamma_i2, x0 = (y_down + y_up)/2, args = (i, tau, xi, c), bounds = [(y_down[i], y_up[i]) for i in range(y_down.shape[0])]).x
        else:
            y_i = np.zeros(0)
        
        # Concatenate values at endpoints
        y_i = np.concatenate([np.zeros(1), y_i, np.zeros(1)], axis=0)
        
        y += [y_i]
    return y

def dytilde_dtau(y, tau, xi, nu):
    """
        Inputs:
            y: list of nu float array of shape (omega_nu[i]+2)
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            nu: int, number of distinct intervals that constitute the support 
        Outputs:
            deriv: list of nu float array of shape (omega_nu[i]+2,p), deriv[i][j,k] = (\partial \tilde{y}_j^i/\partial \tau_k)(tau)
    """
    deriv = []
    for i in range(nu):
        num1 = tau[np.newaxis,:]/((tau[np.newaxis,:] - xi[i][:,np.newaxis])**2 + y[i][:,np.newaxis]**2)
        num2 = tau[np.newaxis,:]**2*(tau[np.newaxis,:] - xi[i][:,np.newaxis])/((tau[np.newaxis,:] - xi[i][:,np.newaxis])**2 + y[i][:,np.newaxis]**2)**2
        den = np.tile((tau[np.newaxis,:]**2*y[i][:,np.newaxis]/((tau[np.newaxis,:] - xi[i][:,np.newaxis])**2 + y[i][:,np.newaxis]**2)**2).sum(axis=1)[:,np.newaxis], (1, num1.shape[1]))

        deriv_i = np.zeros(num1.shape)
        # Select indexes where the denominator is not zero, usually this is [1:-1]
        idx = den != 0
        deriv_i[idx] = (num1 - num2)[idx]/den[idx]
        deriv += [deriv_i]
    return deriv

def dy_dxi(y, tau, xi, nu):
    """
        Inputs:
            y: list of nu float array of shape (omega_nu[i]+2)
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            nu: int, number of distinct intervals that constitute the support 
        Outputs:
            deriv: list of nu float array of shape (omega_nu[i]+2), deriv[i][j,k] = \partial y_j^i/\partial x_j^i
    """
    deriv = []
    for i in range(nu):
        num = (tau[np.newaxis,:]**2*(tau[np.newaxis,:] - xi[i][:,np.newaxis])/((tau[np.newaxis,:] - xi[i][:,np.newaxis])**2 + y[i][:,np.newaxis]**2)**2).sum(axis=1)
        den = (tau[np.newaxis,:]**2*y[i][:,np.newaxis]/((tau[np.newaxis,:] - xi[i][:,np.newaxis])**2 + y[i][:,np.newaxis]**2)**2).sum(axis=1)
        
        deriv_i = np.zeros(num.shape)
        # Select indexes where the denominator is not zero, usually this is [1:-1]
        idx = den != 0
        deriv_i[idx] = num[idx]/den[idx]
        deriv += [deriv_i]
    return deriv

def dy_dtau(y, tau, xi, nu, u_nu, omega_nu):
    """
        Inputs:
            y: list of nu float array of shape (omega_nu[i]+2)
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            nu: int, number of distinct intervals that constitute the support 
            u_nu: array of shape (2*nu), the support endpoints
            omega_nu: float array of shape (nu), omega_nu[i] contains how many population eigenvalues correspond to the support interval [u_nu[2*i], u_nu[2*i+1]]
        Outputs:
            deriv: list of nu float array of shape (omega_nu[i]+2,p), deriv[i][j,k] = \partial y_j^i/\partial \tau_k
    """
    dytildedtau = dytilde_dtau(y, tau, xi, nu)
    dydxi = dy_dxi(y, tau, xi, nu)
    dxidtau = dxi_dtau(nu, u_nu, omega_nu, tau)
    
    deriv = []
    for i in range(nu):
        deriv_i = dytildedtau[i] + dydxi[i][:,np.newaxis]*dxidtau[i]
        deriv += [deriv_i]
    return deriv


# Density of the LSD of th sample eigenvalues

def mLH(zi, tau):
    """
        Inputs:
            z: complex array of shape (L)
            tau: array of shape (p), population eigenvalues
        Outputs:
            m: complex array of shape (L), m[i] = m_LH(zi[i])
    """    
    m = 1 + (zi[:,np.newaxis]/(tau[np.newaxis,:] - zi[:,np.newaxis])).mean(axis=1)
    return m

def compute_z(y, xi, nu):
    """
        Inputs:
            y: list of nu float array of shape (omega_nu[i]+2)
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            nu: int, number of distinct intervals that constitute the support 
        Outputs:
            z: list of nu complex array of shape (omega_nu[i]+2), z[i][j] = xi[i][j] + y[i][j]*1j
    """
    z = []
    for i in range(nu):
        z_i = xi[i] + y[i]*1j
        z += [z_i]  
    return z

def density(z, tau, nu, c):
    """
        Inputs:
            z: list of nu complex array of shape (omega_nu[i]+2), z[i][j] = xi[i][j] + y[i][j]*1j
            tau: array of shape (p), population eigenvalues
            nu: int, number of distinct intervals that constitute the support 
            c: float, p/n
        Outputs:
            x: list of nu float array of shape (omega_nu[i]+2), (x[i][j], f[i][j]) is the mapping in real space of z[i][j] (= xi[i][j] + y[i][j]*1j) which is in u-space
            f: list of nu float array of shape (omega_nu[i]+2), (x[i][j], f[i][j]) is the mapping in real space of z[i][j] (= xi[i][j] + y[i][j]*1j) which is in u-space
    """
    x, f = [], []
    for i in range(nu):
        x_i = z[i] - c*z[i]*mLH(z[i],tau)
        x_i = x_i.real
        f_i = 1/c/np.pi*(-1/z[i]).imag
        
        x += [x_i]
        f += [f_i]  
    
    return x, f

def df_dtau(dxidtau, dydtau, z, y, tau, xi, nu, u_nu, c):
    """
        Inputs:
            dxidtau: list of nu float array of shape (omega_nu[i]+2,p), where deriv[i][j,k] = \partial xi[i][j]/\partial tau[k]
            dydtau: list of nu float array of shape (omega_nu[i]+2, p), deriv[i][j,k] = \partial y_j^i/\partial \tau_k
            z: list of nu complex array of shape (omega_nu[i]+2), z[i][j] = xi[i][j] + y[i][j]*1j
            y: list of nu float array of shape (omega_nu[i]+2)
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            nu: int, number of distinct intervals that constitute the support 
            u_nu: array of shape (2*nu), the support endpoints
            c: float, p/n
        Outputs:
            deriv: list of nu float array of shape (omega_nu[i]+2,p), deriv[i][j,k] = \partial f_j^i/\partial \tau_k
    """
    deriv = []
    for i in range(nu):
        deriv_i = 1/c/np.pi*((dxidtau[i] + dydtau[i]*1j)/z[i][:,np.newaxis]**2).imag
        deriv += [deriv_i]
    return deriv


def ddmLH_ddtau(dxidtau, dydtau, z, y, tau, xi, nu, u_nu):
    """
        Inputs:
            dxidtau: list of nu float array of shape (omega_nu[i]+2,p), where deriv[i][j,k] = \partial xi[i][j]/\partial tau[k]
            dydtau: list of nu float array of shape (omega_nu[i]+2, p), deriv[i][j,k] = \partial y_j^i/\partial \tau_k
            z: list of nu complex array of shape (omega_nu[i]+2), z[i][j] = xi[i][j] + y[i][j]*1j
            y: list of nu float array of shape (omega_nu[i]+2)
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            nu: int, number of distinct intervals that constitute the support 
            u_nu: array of shape (2*nu), the support endpoints
        Outputs:
            deriv: list of nu float array of shape (omega_nu[i]+2,p), deriv[i][j,k] = d{m_LH}_j^i/d\tau_k
    """    
    p = tau.shape[0]
    deriv = []
    for i in range(nu):
        dmLHdtau = -z[i][:,np.newaxis]/(tau[np.newaxis,:] - z[i][:,np.newaxis])**2/p
        dmLHdz = (tau[np.newaxis,:]/(tau[np.newaxis,:] - z[i][:,np.newaxis])**2).mean(axis=1)
        dzdtau = dxidtau[i] + dydtau[i]*1j
        deriv_i = dmLHdtau + dmLHdz[:,np.newaxis] * dzdtau
        deriv += [deriv_i]
    return deriv

def dx_dtau(dxidtau, dydtau, z, y, tau, xi, nu, u_nu, c):
    """
        Inputs:
            dxidtau: list of nu float array of shape (omega_nu[i]+2,p), where deriv[i][j,k] = \partial xi[i][j]/\partial tau[k]
            dydtau: list of nu float array of shape (omega_nu[i]+2, p), deriv[i][j,k] = \partial y_j^i/\partial \tau_k
            z: list of nu complex array of shape (omega_nu[i]+2), z[i][j] = xi[i][j] + y[i][j]*1j
            y: list of nu float array of shape (omega_nu[i]+2)
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            nu: int, number of distinct intervals that constitute the support 
            u_nu: array of shape (2*nu), the support endpoints
            c: float, p/n
        Outputs:
            deriv: list of nu float array of shape (omega_nu[i]+2,p), deriv[i][j,k] = \partial x_j^i/\partial \tau_k
    """    
    ddmLHdtau = ddmLH_ddtau(dxidtau, dydtau, z, y, tau, xi, nu, u_nu)
    deriv = []
    for i in range(nu):
        deriv_i = (dxidtau[i] + dydtau[i]*1j) * (1 - c*mLH(z[i], tau))[:,np.newaxis] - c*z[i][:,np.newaxis]*ddmLHdtau[i]
        deriv += [deriv_i.real]
    return deriv

def dxf_dtau(z, y, tau, xi, nu, u_nu, omega_nu, c):
    """
        Inputs:
            z: list of nu complex array of shape (omega_nu[i]+2), z[i][j] = xi[i][j] + y[i][j]*1j
            y: list of nu float array of shape (omega_nu[i]+2)
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            nu: int, number of distinct intervals that constitute the support 
            u_nu: array of shape (2*nu), the support endpoints
            omega_nu: float array of shape (nu), omega_nu[i] contains how many population eigenvalues correspond to the support interval [u_nu[2*i], u_nu[2*i+1]]
            c: float, p/n
        Outputs:
        c: float, p/n
            dfdtau: list of nu float array of shape (omega_nu[i]+2,p), deriv[i][j,k] = \partial f_j^i/\partial \tau_k
            dxdtau: list of nu float array of shape (omega_nu[i]+2,p), deriv[i][j,k] = \partial x_j^i/\partial \tau_k
    """    
    dxidtau = dxi_dtau(nu, u_nu, omega_nu, tau)
    dydtau = dy_dtau(y, tau, xi, nu, u_nu, omega_nu)
    
    dfdtau = df_dtau(dxidtau, dydtau, z, y, tau, xi, nu, u_nu, c)
    dxdtau = dx_dtau(dxidtau, dydtau, z, y, tau, xi, nu, u_nu, c)
    
    return dxdtau, dfdtau
    

# Cumulative Distribution Function

def cdf(x, f, c, p, omega_nu, nu):
    """
        Inputs:
            x: list of nu float array of shape (omega_nu[i]+2), (x[i][j], f[i][j]) is the mapping in real space of z[i][j] (= xi[i][j] + y[i][j]*1j) which is in u-space
            f: list of nu float array of shape (omega_nu[i]+2), f[i][j] spectral density at x[i][j]
            c: float, p/n
            p: int, dimension
            tau: array of shape (p), population eigenvalues
            omega_nu: float array of shape (nu), omega_nu[k] contains how many population eigenvalues correspond to the support interval [u_nu[2*k], u_nu[2*k+1]]
            nu: int, number of distinct intervals that constitute the support 
        Outputs:
            x_tilde: list of nu+1 float array of shape (omega_nu[i]+2), concatenation of [np.zeros(1)] and x
            F: list of nu+1 float array of shape (omega_nu[i]+2), F[i][j] spectral cdf at x_tilde[i][j]
            tilde_F: list of nu+1 float array of shape (omega_nu[i]+1), tilde_F[i][j] the uncorrected spectral cdf at x[i][j+1]
    """     
    x_tilde = [np.zeros(1)] + x
    F = [np.maximum(0,(1-1/c)*np.ones(1))]
    tilde_F = [np.maximum(0,(1-1/c)*np.ones(1))]
    
    for i in range(nu):
        F_i = np.zeros(omega_nu[i]+2)
        # Interval endpoints' formulas
        F_i[0] = F[-1][-1]
        F_i[-1] = F[0][0] + (1-F[0][0])*omega_nu[:i+1].sum()/p  # /!\ Here we differ from the article that suggests F_i[-1] = omega_nu[:i+1].sum()/p, however that does not seem to respect F[-1][-1] = 1 when c > 1
        
        tilde_F_i = F_i[0] + 1/2*((x[i][1:] - x[i][:-1])*(f[i][1:] + f[i][:-1])).cumsum()
        if tilde_F_i[-1] > F_i[0]:
            F_i[1:] = F_i[0] + (tilde_F_i - F_i[0])*(F_i[-1] - F_i[0])/(tilde_F_i[-1] - F_i[0])
        else:
            F_i[1:] = F_i[0]
        
        F += [F_i]
        tilde_F += [tilde_F_i]
    
    return x_tilde, F, tilde_F

def dtildeF_dtau(z, y, tau, xi, nu, u_nu, omega_nu, c):
    """
        Inputs:
            z: list of nu complex array of shape (omega_nu[i]+2), z[i][j] = xi[i][j] + y[i][j]*1j
            y: list of nu float array of shape (omega_nu[i]+2)
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            nu: int, number of distinct intervals that constitute the support 
            u_nu: array of shape (2*nu), the support endpoints
            omega_nu: float array of shape (nu), omega_nu[i] contains how many population eigenvalues correspond to the support interval [u_nu[2*i], u_nu[2*i+1]]
            c: float, p/n
        Outputs:
            deriv: list of nu+1 float array of shape (omega_nu[i]+1,p), deriv[i][j,k] = (\partial tilde_F_j^i/\partial tau_k)
    """    
    x, f = density(z, tau, nu, c)
    dxdtau, dfdtau = dxf_dtau(z, y, tau, xi, nu, u_nu, omega_nu, c)
    
    deriv = [np.zeros((1,tau.shape[0]))]
    for i in range(nu):
        deriv_i1 = ((dxdtau[i][1:,:] - dxdtau[i][:-1,:])*(f[i][1:,np.newaxis] + f[i][:-1,np.newaxis])/2).cumsum(axis=0)
        deriv_i2 = ((dfdtau[i][1:,:] + dfdtau[i][:-1,:])*(x[i][1:,np.newaxis] - x[i][:-1,np.newaxis])/2).cumsum(axis=0)
        
        deriv += [deriv_i1 + deriv_i2]
        
    return deriv

def dF_dtau(F, tilde_F, z, y, tau, xi, nu, omega_nu, u_nu, c):
    """
        Inputs:
            F: list of nu+1 float array of shape (omega_nu[i]+2), F[i][j] spectral cdf at x_tilde[i][j]
            tilde_F: list of nu+1 float array of shape (omega_nu[i]+1), tilde_F[i][j] the uncorrected spectral cdf at x[i][j+1]
            z: list of nu complex array of shape (omega_nu[i]+2), z[i][j] = xi[i][j] + y[i][j]*1j
            y: list of nu float array of shape (omega_nu[i]+2)
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            nu: int, number of distinct intervals that constitute the support 
            omega_nu: float array of shape (nu), omega_nu[k] contains how many population eigenvalues correspond to the support interval [u_nu[2*k], u_nu[2*k+1]]
            u_nu: array of shape (2*nu), the support endpoints
            c: float, p/n
        Outputs:
            deriv: list of nu+1 float array of shape (omega_nu[i]+2,p), deriv[i][j,k] = (\partial tilde_F_j^i/\partial tau_k)
    """    
    p = tau.shape[0]
    
    dtildeFdtau = dtildeF_dtau(z, y, tau, xi, nu, u_nu, omega_nu, c)
    
    deriv = [np.zeros((1,p))]
    for i in range(1,nu+1):
        deriv_i = np.zeros((omega_nu[i-1]+2, p))
        deriv_i1 = np.zeros((omega_nu[i-1]+2, p))
        deriv_i2 = np.zeros((omega_nu[i-1]+2, p))
        
        if F[i][-1] > F[i][0]:
            deriv_i1[1:,:] = (F[i][-1] - F[i][0])*(dtildeFdtau[i] - F[i][0])/(tilde_F[i][-1] - F[i][0])
            deriv_i2[1:,:] = (F[i][-1] - F[i][0])*dtildeFdtau[i][-1:,:]*(tilde_F[i][:,np.newaxis] - F[i][0])/(tilde_F[i][-1] - F[i][0])**2
        
        deriv_i = deriv_i1 - deriv_i2
        deriv_i[0,:] = deriv[-1][-1,:]
        deriv += [deriv_i]
    return deriv


# Discretization of the Sample Spectral CDF

def sample_eigenvalues(x_tilde, F, nu, p, n):
    """
        Inputs:
            x_tilde: list of nu+1 float array of shape (omega_nu[i]+2), concatenation of [np.zeros(1)] and x
            F: list of nu+1 float array of shape (omega_nu[i]+2), F[i][j] spectral cdf at x_tilde[i][j]
            nu: int, number of distinct intervals that constitute the support 
            p: int, the dimension
            n: int, the number of samples
        Outputs:
            lambda_: float array of shape (p), sample eigenvalues
            X: float array of shape (p+1), a \mapsto X(a) a piecewise linear approximaton of \int_F(0)^a F^{-1}(x)dx, evaluated on a grid [0,1/p,...,(p-1)/p]
            kappa: list of length nu+1 of int array of variable shapes (1D), indexes from F^{-1}
            j_kappa: list of length nu+1 of int array of variable shapes (same shapes as kappa), the corresponding j-indexes of kappa in each interval
    """    
    lambda_ = np.zeros(p)
    X = np.zeros(p+1)  
    kappa = [(np.ones(max(0,p-n)).cumsum()-1).astype(int)]
    j_kappa = [np.zeros(max(0,p-n)).astype(int)]
    
    for i in range(1,nu+1):
        # The indexes kappa
        kappa_i = (np.ones(int(np.floor(p*F[i][-1]) - np.ceil(p*F[i][0]) + 1)).cumsum() - 1) + np.ceil(p*F[i][0])
        # Remove the final equality case
        kappa_i = kappa_i[kappa_i < p*F[i][-1]]
        kappa_i = kappa_i.astype(int)
        
        # The corresponding j-indexes in the interval
        j_kappa_i = np.argmax((p*F[i][:-1,np.newaxis] <= kappa_i[np.newaxis,:]) & (p*F[i][1:,np.newaxis] > kappa_i[np.newaxis,:]), axis=0).astype(int)

        # Easy way: lambda_[kappa] = x_tilde[i][j_kappa]
        # Hard way: using a \mapsto X(a) a piecewise linear approximaton of \int_F(0)^a F^{-1}(x)dx
        cumsum_term = np.concatenate([np.zeros(1), ((F[i][1:] - F[i][:-1])*(x_tilde[i][1:] + x_tilde[i][:-1])/2).cumsum()])
        X[kappa_i] = cumsum_term[j_kappa_i] + (kappa_i/p - F[i][j_kappa_i])*x_tilde[i][j_kappa_i] + (kappa_i/p - F[i][j_kappa_i])**2*(x_tilde[i][j_kappa_i+1] - x_tilde[i][j_kappa_i])/(F[i][j_kappa_i+1] - F[i][j_kappa_i])/2
        
        kappa += [kappa_i]
        j_kappa += [j_kappa_i]
    
    # Compute the final point X[-1]
    # There is more complex method taking in account the approx until X[-2]
    for i in range(1,nu+1):
        X[-1] += ((F[i][1:] - F[i][:-1])*(x_tilde[i][1:] + x_tilde[i][:-1])/2).sum()
    
    lambda_ = np.diff(X, axis=0)*p
    
    return lambda_, X, kappa, j_kappa

def dX_dtau(lambda_, X, kappa, j_kappa, x_tilde, F, tilde_F, z, y, tau, xi, nu, omega_nu, u_nu, c, p, n):
    """
        Inputs:
            lambda_: float array of shape (p), sample eigenvalues
            X: float array of shape (p+1), a \mapsto X(a) a piecewise linear approximaton of \int_F(0)^a F^{-1}(x)dx, evaluated on a grid [0,1/p,...,(p-1)/p]
            kappa: list of length nu+1 of int array of variable shapes (1D), indexes from F^{-1}
            j_kappa: list of length nu+1 of int array of variable shapes (same shapes as kappa), the corresponding j-indexes of kappa in each interval
            x_tilde: list of nu+1 float array of shape (omega_nu[i]+2), concatenation of [np.zeros(1)] and x
            F: list of nu+1 float array of shape (omega_nu[i]+2), F[i][j] spectral cdf at x_tilde[i][j]
            tilde_F: list of nu float array of shape (omega_nu[i]+1), tilde_F[i][j] the uncorrected spectral cdf at x[i][j+1]
            z: list of nu complex array of shape (omega_nu[i]+2), z[i][j] = xi[i][j] + y[i][j]*1j
            y: list of nu float array of shape (omega_nu[i]+2)
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            nu: int, number of distinct intervals that constitute the support 
            omega_nu: float array of shape (nu), omega_nu[k] contains how many population eigenvalues correspond to the support interval [u_nu[2*k], u_nu[2*k+1]]
            u_nu: array of shape (2*nu), the support endpoints
            c: float, p/n
            p: int, the dimension
            n: int, the number of samples
        Outputs:
            deriv: float array of shape (p+1, p), deriv[j,k] = (\partial X/\partial tau_k)(j/p)
    """    
    dFdtau = dF_dtau(F, tilde_F, z, y, tau, xi, nu, omega_nu, u_nu, c)
    dxdtau, dfdtau = dxf_dtau(z, y, tau, xi, nu, u_nu, omega_nu, c)
    dxdtau = [[np.zeros((1,p))]] + dxdtau
    dfdtau = [[np.zeros((1,p))]] + dfdtau
    
    deriv = np.zeros((p+1,p))
    for i in range(1,nu+1):
        term1 = (dFdtau[i][1:,:] - dFdtau[i][:-1,:])*(x_tilde[i][:-1,np.newaxis] + x_tilde[i][1:,np.newaxis])/2 + (dxdtau[i][1:,:] + dxdtau[i][:-1,:])*(F[i][1:,np.newaxis] - F[i][:-1,np.newaxis])/2
        term1 = np.concatenate([np.zeros((1,p)), term1.cumsum(axis=0)], axis=0)[j_kappa[i]]
        
        term2_1 = (kappa[i][:,np.newaxis]/p - F[i][j_kappa[i],np.newaxis])*dxdtau[i][j_kappa[i],:]
        term2_2 = - dFdtau[i][j_kappa[i],:]*x_tilde[i][j_kappa[i],np.newaxis]
        term2_3 = - dFdtau[i][j_kappa[i],:]*(kappa[i][:,np.newaxis]/p - F[i][j_kappa[i],np.newaxis])**2*(x_tilde[i][j_kappa[i]+1,np.newaxis] - x_tilde[i][j_kappa[i],np.newaxis])/(F[i][j_kappa[i]+1,np.newaxis] - F[i][j_kappa[i],np.newaxis])
        term2_4 = (kappa[i][:,np.newaxis]/p - F[i][j_kappa[i],np.newaxis])**2*(dxdtau[i][j_kappa[i]+1,:] - dxdtau[i][j_kappa[i],:])/2/(F[i][j_kappa[i]+1,np.newaxis] - F[i][j_kappa[i],np.newaxis])
        term2_5 = - (dFdtau[i][j_kappa[i]+1,:] - dFdtau[i][j_kappa[i],:])*(kappa[i][:,np.newaxis]/p - F[i][j_kappa[i],np.newaxis])**2*(x_tilde[i][j_kappa[i]+1,np.newaxis] - x_tilde[i][j_kappa[i],np.newaxis])/2/(F[i][j_kappa[i]+1,np.newaxis] - F[i][j_kappa[i],np.newaxis])
        term2 = term2_1 + term2_2 + term2_3 + term2_4 + term2_5
        
        deriv_i = term1 + term2
        
        deriv[kappa[i]] = deriv_i
    
    # Compute the final point X[-1]
    # There is more complex method taking in account the approx until X[-2]
    for i in range(1,nu+1):
        deriv[-1] += ((dFdtau[i][1:,:] - dFdtau[i][:-1,:])*(x_tilde[i][:-1,np.newaxis] + x_tilde[i][1:,np.newaxis])/2 + (dxdtau[i][1:,:] + dxdtau[i][:-1,:])*(F[i][1:,np.newaxis] - F[i][:-1,np.newaxis])/2).sum(axis=0)
    
    return deriv

def dlambda_dtau(lambda_, X, kappa, j_kappa, x_tilde, F, tilde_F, z, y, tau, xi, nu, omega_nu, u_nu, c, p, n):
    """
        Inputs:
            lambda_: float array of shape (p), sample eigenvalues
            X: float array of shape (p), a \mapsto X(a) a piecewise linear approximaton of \int_F(0)^a F^{-1}(x)dx, evaluated on a grid [0,1/p,...,(p-1)/p]
            kappa: list of length nu+1 of int array of variable shapes (1D), indexes from F^{-1}
            j_kappa: list of length nu+1 of int array of variable shapes (same shapes as kappa), the corresponding j-indexes of kappa in each interval
            x_tilde: list of nu+1 float array of shape (omega_nu[i]+2), concatenation of [np.zeros(1)] and x
            F: list of nu+1 float array of shape (omega_nu[i]+2), F[i][j] spectral cdf at x_tilde[i][j]
            tilde_F: list of nu float array of shape (omega_nu[i]+1), tilde_F[i][j] the uncorrected spectral cdf at x[i][j+1]
            z: list of nu complex array of shape (omega_nu[i]+2), z[i][j] = xi[i][j] + y[i][j]*1j
            y: list of nu float array of shape (omega_nu[i]+2)
            tau: array of shape (p), population eigenvalues
            xi: list of nu float array of shape (omega_nu[i] + 2), xi[i] contains the grid that covers the interval [u_nu[2*i], u_nu[2*i+1]]
            nu: int, number of distinct intervals that constitute the support 
            omega_nu: float array of shape (nu), omega_nu[k] contains how many population eigenvalues correspond to the support interval [u_nu[2*k], u_nu[2*k+1]]
            u_nu: array of shape (2*nu), the support endpoints
            c: float, p/n
            p: int, the dimension
            n: int, the number of samples
        Outputs:
            deriv: float array of shape (p, p), deriv[j,k] = \partial \lambda_j/\partial tau_k
    """    
    dXdtau = dX_dtau(lambda_, X, kappa, j_kappa, x_tilde, F, tilde_F, z, y, tau, xi, nu, omega_nu, u_nu, c, p, n)
    deriv = p*np.diff(dXdtau, axis=0)
    return deriv


# QuEST function

def QuEST(tau, p, n):
    """
        Inputs:
            tau: array of shape (p), population eigenvalues
            p: int, the dimension
            n: int, the number of samples
        Outputs:
            lambda_: float array of shape (p), sample eigenvalues
            dlambdadtau: float array of shape (p, p), deriv[j,k] = \partial \lambda_j/\partial tau_k
            x: list of nu float array of shape (omega_nu[i]+2), (x[i][j], f[i][j]) is the mapping in real space of z[i][j] (= xi[i][j] + y[i][j]*1j) which is in u-space
            f: list of nu float array of shape (omega_nu[i]+2), (x[i][j], f[i][j]) is the mapping in real space of z[i][j] (= xi[i][j] + y[i][j]*1j) which is in u-space
            x_tilde: list of nu+1 float array of shape (omega_nu[i]+2), concatenation of [np.zeros(1)] and x
            F: list of nu+1 float array of shape (omega_nu[i]+2), F[i][j] spectral cdf at x_tilde[i][j]
    """    
    c = p/n
    
    # Remember the order and sort
    tau = np.maximum(tau,0)
    idx = np.argsort(tau)
    tau = np.sort(tau)
    
    # Compute the QuEST function
    nu, u_nu, omega_nu, w, t = support(tau, c)
    xi = grid(nu, u_nu, omega_nu)
    y = find_y(w, t, tau, nu, xi, c)
    z = compute_z(y, xi, nu)
    x,f = density(z, tau, nu, c)
    x_tilde, F, tilde_F = cdf(x, f, c, p, omega_nu, nu)
    lambda_, X, kappa, j_kappa = sample_eigenvalues(x_tilde, F, nu, p, n)
    dlambdadtau = dlambda_dtau(lambda_, X, kappa, j_kappa, x_tilde, F, tilde_F, z, y, tau, xi, nu, omega_nu, u_nu, c, p, n)
    
    # Come back to the original order
    lambda_unsrt, dlambdadtau_unsrt = np.zeros(p), np.zeros((p,p))
    lambda_unsrt[idx] = lambda_
    dlambdadtau_unsrt[idx] = dlambdadtau
    dlambdadtau_unsrt[:,idx] = dlambdadtau_unsrt
        
    return lambda_unsrt, dlambdadtau_unsrt, x, f, x_tilde, F


# Test: Sample PDF

def sample_pdf(tau, p, n, n_mc = 100, bins = 50):
    lambda_q, dlambdadtau_, x, f, x_tilde, F = QuEST(tau, p, n)
    
    plt.figure()
    plt.hist(lambda_q, bins=bins)
    plt.plot(x[0], f[0]*3*p/bins)
    plt.show()
    
    plt.figure()
    for i in range(p):
        plt.plot(lambda_q[i]*np.ones(x_tilde[1].shape[0]), np.linspace(0,(i+1/2)/p,x_tilde[1].shape[0]))
        plt.plot(np.linspace(0,lambda_q[i],x_tilde[1].shape[0]), (i+1/2)/p*np.ones(x_tilde[1].shape[0]))
    plt.plot(x_tilde[1], F[1])
    plt.show()
    
    lambda_ = np.zeros(0)
    for i in range(n_mc):
        X_sample = np.random.normal(size=(n,p)) @ np.sqrt(np.diag(tau))
        sample_cov = X_sample.T @ X_sample/n
        lambda_sample, _ = sp.linalg.eigh(sample_cov)
        lambda_ = np.concatenate([lambda_, lambda_sample])
        
    plt.figure()
    plt.hist(lambda_, bins=bins)
    plt.plot(x[0], f[0]*3*p/bins*n_mc)
    plt.show()
    
    return x, f, x_tilde, F, lambda_q, lambda_sample

# Test: Euler Jacobian

def jacobian(tau, p, n, eps = 1e-3):
    c = p/n
    
    nu, u_nu, omega_nu, w, t = support(tau, c)
    xi = grid(nu, u_nu, omega_nu)
    y = find_y(w, t, tau, nu, xi, c)
    z = compute_z(y, xi, nu)
    mlh = []
    for i in range(nu):
        mlh += [mLH(z[i], tau)]
    x,f = density(z, tau, nu, c)
    x_tilde, F, tilde_F = cdf(x, f, c, p, omega_nu, nu)
    lambda_, X, kappa, j_kappa = sample_eigenvalues(x_tilde, F, nu, p, n)
    
    jac_u_a = du_dtau(u_nu, tau)
    jac_xi_a = dxi_dtau(nu, u_nu, omega_nu, tau)
    jac_y_a = dy_dtau(y, tau, xi, nu, u_nu, omega_nu)
    jac_mlh_a = ddmLH_ddtau(jac_xi_a, jac_y_a, z, y, tau, xi, nu, u_nu)
    jac_x_a, jac_f_a = dxf_dtau(z, y, tau, xi, nu, u_nu, omega_nu, c)
    jac_F_a = dF_dtau(F, tilde_F, z, y, tau, xi, nu, omega_nu, u_nu, c)
    jac_X_a = dX_dtau(lambda_, X, kappa, j_kappa, x_tilde, F, tilde_F, z, y, tau, xi, nu, omega_nu, u_nu, c, p, n)
    jac_tau_a = dlambda_dtau(lambda_, X, kappa, j_kappa, x_tilde, F, tilde_F, z, y, tau, xi, nu, omega_nu, u_nu, c, p, n)
    
    jac_u = np.zeros(jac_u_a.shape)
    jac_xi = []
    jac_y = []
    jac_mlh = []
    jac_x = []
    jac_f = []
    jac_F = []
    for i in range(nu):
        jac_xi += [np.zeros(jac_xi_a[i].shape)]
        jac_y += [np.zeros(jac_y_a[i].shape)]
        jac_mlh += [np.zeros(jac_mlh_a[i].shape, dtype=np.complex128)]
        jac_x += [np.zeros(jac_x_a[i].shape)]
        jac_f += [np.zeros(jac_f_a[i].shape)]
        jac_F += [np.zeros(jac_F_a[i].shape)]
    jac_F += [np.zeros(jac_F_a[nu].shape)]
    jac_X = np.zeros((p+1,p))
    jac_tau = np.zeros((p,p))
    
    for i in range(p):
        perturb = np.zeros(p)
        perturb[i] = eps
        tau1 = tau + perturb
        
        nu1, u_nu1, omega_nu1, w1, t1 = support(tau1, c)
        xi1 = grid(nu1, u_nu1, omega_nu1)
        y1 = find_y(w1, t1, tau1, nu1, xi1, c)
        z1 = compute_z(y1, xi1, nu1)
        mlh1 = []
        for j in range(nu1):
            mlh1 += [mLH(z1[j], tau1)] 
        x1,f1 = density(z1, tau1, nu1, c)
        x_tilde1, F1, tilde_F1 = cdf(x1, f1, c, p, omega_nu1, nu1)
        lambda1, X1, kappa1, j_kappa1 = sample_eigenvalues(x_tilde1, F1, nu1, p, n)
        
        jac_u[:,i] = (u_nu1 - u_nu)/eps
        if nu != nu1:
            print("nu and nu1 are different for i = ", str(i))
        else:
            for j in range(nu):
                jac_xi[j][:,i] = (xi1[j] - xi[j])/eps
                jac_y[j][:,i] = (y1[j] - y[j])/eps
                jac_mlh[j][:,i] = (mlh1[j] - mlh[j])/eps
                jac_x[j][:,i] = (x1[j] - x[j])/eps
                jac_f[j][:,i] = (f1[j] - f[j])/eps
                jac_F[j][:,i] = (F1[j] - F[j])/eps
            jac_F[nu][:,i] = (F1[nu] - F[nu])/eps
        jac_X[:,i] = (X1 - X)/eps
        jac_tau[:,i] = (lambda1 - lambda_)/eps
        
        
    return jac_u_a, jac_xi_a, jac_y_a, jac_mlh_a, jac_x_a, jac_f_a, jac_F_a, jac_X_a, jac_tau_a, jac_u, jac_xi, jac_y, jac_mlh, jac_x, jac_f, jac_F, jac_X, jac_tau


# Numerical inversion of QuEST function

class compute_loss_wasserstein:
    def __init__(self, power = 1, regu = 1):
        self.computed = False
        self.power = power
        self.regu = regu
        return
    
    def loss(self, tau, train_lambda_, p, n):
        lambda_, dlambdadtau, _, _, _, _ = QuEST(tau, p, n)
        self.grad = dlambdadtau
        self.lambda_ = lambda_
        self.computed = True
        
        # Inspired from Scipy 1D Wasserstein distance: https://github.com/scipy/scipy/blob/v1.11.4/scipy/stats/_stats_py.py#L9733-L9807
        u_values, v_values = lambda_, train_lambda_
        
        u_weights = np.ones(u_values.shape)
        u_weights = u_weights / u_weights.sum(axis = 0)
        
        v_weights = torch.ones(v_values.shape)
        v_weights = v_weights / v_weights.sum(axis = 0)
        
        u_sorter = np.argsort(u_values, axis=0)
        v_sorter = np.argsort(v_values, axis=0)
        
        all_values = np.concatenate((u_values, v_values), axis=0)
        all_sorter = all_values.argsort(axis=0)
        all_values_sort = all_values[all_sorter]
        
        # Remember for grad computing
        self.all_sorter = all_sorter
    
        # Compute the differences between pairs of successive values of u and v.
        deltas = np.diff(all_values_sort, axis=0)
    
        # Get the respective positions of the values of u and v among the values of
        # both distributions.
        u_values_sort = u_values[u_sorter]
        v_values_sort = v_values[v_sorter]
        
        u_cdf_indices = all_values_sort[:-1].searchsorted(u_values_sort, 'right')
        v_cdf_indices = all_values_sort[:-1].searchsorted(v_values_sort, 'right')
        
        # Calculate the CDFs of u and v using their weights, if specified.
        u_cdf = np.zeros(all_values.shape)
        v_cdf = np.zeros(all_values.shape)
        
        u_cdf[u_cdf_indices] = u_weights[u_sorter]
        u_cdf = u_cdf[1:].cumsum(axis=0)
        u_cdf = u_cdf / u_cdf[-1]
        
        v_cdf[v_cdf_indices] = v_weights[v_sorter]
        v_cdf = v_cdf[1:].cumsum(axis=0)
        v_cdf = v_cdf / v_cdf[-1]
        
        # Remember for grad computing
        self.u_cdf = u_cdf
        self.v_cdf = v_cdf
        
        # We do not normalize the power by 1/p at the end, to make it differentiable
        distance = ((np.abs(u_cdf - v_cdf)**self.power * deltas)).sum(axis=0) + self.regu*(tau - train_lambda_).mean()**2
        return distance
    
    def jacobian(self, tau, train_lambda_, p, n):
        if self.computed:
            self.computed = False
        else:
            l = self.loss(tau, train_lambda_, p, n)
        
        all_grad = np.concatenate((self.grad, np.zeros((p, p))), axis=0)
        all_grad_sort = all_grad[self.all_sorter]
        grad_deltas = np.diff(all_grad_sort, axis=0)
        
        g = ((np.abs(self.u_cdf - self.v_cdf)[:,None]**self.power * grad_deltas)).sum(axis=0) + self.regu*2*(tau - train_lambda_)[:,np.newaxis].mean(axis=1)
        return g

class compute_loss_l2:
    def __init__(self, regu = 1, power = 2):
        self.computed = False
        self.regu = regu
        return
    
    def loss(self, tau, train_lambda_, p, n):
        lambda_, dlambdadtau, _, _, _, _ = QuEST(tau, p, n)
        self.grad = dlambdadtau
        self.lambda_ = lambda_
        self.computed = True
        l = ((lambda_ - train_lambda_)**2).mean() + self.regu*(lambda_ - train_lambda_).mean()**2 
        return l
    
    def jacobian(self, tau, train_lambda_, p, n):
        if self.computed:
            self.computed = False
            return 2*(self.grad*(self.lambda_ - train_lambda_)[:,np.newaxis]).mean(axis=1)
        else:
            lambda_, dlambdadtau, _, _, _, _ = QuEST(tau, p, n)
            self.grad = dlambdadtau
            self.lambda_ = lambda_
            return 2*(self.grad*(self.lambda_ - train_lambda_)[:,np.newaxis]).mean(axis=1) + self.regu*2*self.grad.mean(axis=1)*(self.lambda_ - train_lambda_)[:,np.newaxis].mean(axis=1)
    
    def hessian(self, tau, train_lambda_, p, n, eps = 1e-3):
        hess = np.zeros((p,p))
        jac = self.jacobian(tau, train_lambda_, p, n)
        
        for i in range(p):
            perturb = np.zeros(p)
            perturb[i] = eps
            tau1 = tau + perturb
            jac1 = self.jacobian(tau1, train_lambda_, p, n)
            hess[:,i] = (jac1 - jac)/eps
        
        return hess
    
class compute_loss_composite:
    def __init__(self, regu = 1, power = 2):
        self.computed = False
        self.regu = regu
        self.power = power
        self.cll = compute_loss_l2(regu = regu, power = power)
        self.clw = compute_loss_wasserstein(regu = regu, power = power)
        return
    
    def loss(self, tau, train_lambda_, p, n):
        ll = self.cll.loss(tau, train_lambda_, p, n)
        lw = self.clw.loss(tau, train_lambda_, p, n)
        return ll + lw
    
    def jacobian(self, tau, train_lambda_, p, n):
        jl = self.cll.jacobian(tau, train_lambda_, p, n)
        jw = self.clw.jacobian(tau, train_lambda_, p, n)
        return jl + jw


def GD(tau_init, train_lambda_, p, n, cl, max_iter = 1000, lr = 3e-2, early_stopping = True, verbose = True):
    """
        Gradient Descent optimizer
    """
    if verbose:
        print("GD algorithm, max_iter =", max_iter, ", lr =", lr)
        
    # Initialization
    tau = tau_init
    loss = cl.loss(tau, train_lambda_, p, n)
    best_tau = tau_init
    best_loss = loss
    best_iter = 0
    
    # Optimization
    for i in range(max_iter):
        if verbose and i % (max_iter//20) == 0:
            print("Loss epoch", i, ":", loss)
            
        # Gradient descent
        dl = cl.jacobian(tau, train_lambda_, p, n)
        tau = tau - lr * dl
        tau = np.sort(tau)
        
        # Remember the best tau
        loss = cl.loss(tau, train_lambda_, p, n)
        if loss < best_loss:
            best_tau = tau
            best_loss = loss
            best_iter = i
        
        # Early stopping
        if early_stopping and i - best_iter > max_iter // 20:
            lr /= 2
        if early_stopping and i - best_iter > max_iter // 10:
            break
        
    tau = best_tau
    if verbose:
        print("Final loss epoch", max_iter, ":", cl.loss(tau, train_lambda_, p, n))
    return tau

def Adam(tau_init, train_lambda_, p, n, cl, max_iter = 1000, lr = 1e-2, betas = (0.9,0.999), eps = 1e-8, w = 0, early_stopping = True, verbose = True):
    """
        Adam optimizer
    """
    if verbose:
        print("Adam algorithm, max_iter =", max_iter, ", lr =", lr, ", betas =", betas)
    
    # Initialization
    tau = tau_init
    loss = cl.loss(tau, train_lambda_, p, n)
    best_tau = tau_init
    best_loss = loss
    best_iter = 0
    m, v = 0, 0
    
    # Optimization
    for i in range(max_iter):
        if verbose and i % (max_iter//20) == 0:
            print("Loss epoch", i, ":", loss)
            
        # Gradient descent
        dl = cl.jacobian(tau, train_lambda_, p, n)
        m = betas[0]*m + (1-betas[0])*dl
        v = betas[1]*v + (1-betas[1])*dl**2
        hat_m = m/(1-betas[0])
        hat_v = v/(1-betas[1])
        tau = tau - lr * hat_m / (np.sqrt(hat_v) + eps)
        tau = np.sort(tau)
        
        # Remember the best tau
        loss = cl.loss(tau, train_lambda_, p, n)
        if loss < best_loss:
            best_tau = tau
            best_loss = loss
            best_iter = i
        
        # Early stopping
        if early_stopping and i - best_iter > max_iter // 20:
            tau = best_tau
            loss = cl.loss(tau, train_lambda_, p, n)
            lr /= 2
        if early_stopping and i - best_iter > max_iter // 10:
            break
        
    tau = best_tau
    if verbose:
        print("Final loss", max_iter, ":", best_loss)
    return tau

def RMSProp(tau_init, train_lambda_, p, n, cl, max_iter = 1000, lr = 1e-2, momentum = 0., alpha = 0.99, eps = 1e-8, centered = False, w = 0, early_stopping = True, verbose = True):
    """
        RMSProp optimizer
    """
    if verbose:
        print("RMSProp algorithm, max_iter =", max_iter, ", lr =", lr, ", momentum =", momentum, ", alpha =", alpha, ", centered =", centered)
    
    # Initialization
    tau = tau_init
    loss = cl.loss(tau, train_lambda_, p, n)
    best_tau = tau_init
    best_loss = loss
    best_iter = 0
    g, v, g_bar, b = 0, 0, 0, 0
    
    # Optimization
    for i in range(max_iter):
        if verbose and i % (max_iter//20) == 0:
            print("Loss epoch", i, ":", loss)
            
        # Gradient descent
        dl = cl.jacobian(tau, train_lambda_, p, n)
        g = dl + w * tau
        v = alpha * v + (1 - alpha) * g**2
        if centered:
            g_bar = alpha*g_bar + (1-alpha)*g
            v  = v - g_bar**2
        if momentum > 0:
            b = momentum * b + g/(np.sqrt(v) + eps)
        else:
            b = lr * g/(np.sqrt(v) + eps)
        if i < 10:
            tau = tau - lr * g
        else:
            tau = tau - lr * g/(np.sqrt(v) + eps)
        tau = np.sort(tau)
        
        # Remember the best tau
        loss = cl.loss(tau, train_lambda_, p, n)
        if loss < best_loss:
            best_tau = tau
            best_loss = loss
            best_iter = i
        
        # Early stopping
        if early_stopping and i - best_iter > max_iter // 20:
            lr /= 2
        if early_stopping and i - best_iter > max_iter // 10:
            break
        
    tau = best_tau
    if verbose:
        print("Final loss", max_iter, ":", best_loss)
    return tau
    
def QuEST_minimization(X_sample, tau_init, method = None, dist = "wass", power = 2, regu = 1, assume_centered = False, tol = 1e-5, max_iter = 1000, lr = 1e-2, betas = (0.9,0.999), momentum = 0., alpha = 0.9, eps = 1e-8, w = 0, centered = False, early_stopping = True, verbose = True):
    """
        Inputs:
            X_sample: float array of sahpe (n, p), with n the number of samples, and p the dimension
            tau_init: float array of shape (p), initial guess of population spectrum
            method: string, optimizer to use, in ['GD', 'Adam', 'composite', 'RMSProp', 'default', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
                * Need callable Jacobian: CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr
                * Need callable Hessian: Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr
                * In the orginial Matlab implementation: we can chose between:
                    * TOMLAB/SNOPT, or
                    * Matlab optimizers: fsolve ('trust-region-dogleg' (default), 'trust-region', or 'levenberg-marquardt'), fmincon ('interior-point' (default), 'trust-region-reflective', 'sqp', 'sqp-legacy', 'active-set')
            dist: string, in ['wass', 'l2']
            power: float, power norm
            regu: float, rgularization term on the mean of the distribution
            assume_centered: bool, if False, X_sample will be demeaned
            tol: float, tolerance used in the optimizer
            max_iter: int, max number of iterations in the optimizer
            lr: float, learning rate for GD and Adam
            betas: tuple of float (beta1, beta2), for Adam optimizer
            momentum: float, for RMSProp
            alpha: float, for RMSProp
            eps: float, for Adam optimizer
            w: float, for Adam optimizer
            centered: bool, for RMSProp
            early_stopping: bool, use of early stopping strategies
            verbose: bool
        Outputs:
            tau: array of shape (p), the estimated population eigenvalues
    """
    n, p = X_sample.shape
    c = p/n
    
    if dist == 'l2':
        cl = compute_loss_l2(regu = regu, power = power)
    elif dist == "wass":
        cl = compute_loss_wasserstein(regu = regu, power = power)
    elif dist == 'composite':
        cl = compute_loss_composite(regu = regu, power = power)
    else:
        raise ValueError(dist+" distance not implemented.")
    
    if assume_centered:
        sample_cov = X_sample.T @ X_sample/n
    else:
        X_mean = X_sample.mean(axis=0)[np.newaxis,:]
        sample_cov = (X_sample - X_mean).T @ (X_sample - X_mean)/(n-1)
    train_lambda_, V = sp.linalg.eigh(sample_cov)
    
    if method == 'GD':
        # Minimization by GD
        tau = GD(tau_init, train_lambda_, p, n, cl, max_iter = max_iter, lr = lr, verbose = verbose)
    
    elif method == None or method == "Adam":
        # Minimization by Adam
        tau = Adam(tau_init, train_lambda_, p, n, cl, max_iter = max_iter, lr = lr, betas = betas, eps = eps, w = w, verbose = verbose)
    
    elif method == "composite":
        # Minimization by Adam
        tau_adam = Adam(tau_init, train_lambda_, p, n, cl, max_iter = max_iter, lr = lr, betas = betas, eps = eps, w = w, verbose = verbose)
        # Then minimization by GD
        tau = GD(tau_adam, train_lambda_, p, n, cl, max_iter = max_iter, lr = lr, verbose = verbose)
    
    elif method == "RMSProp":
        # Minimization by Adam
        tau = RMSProp(tau_init, train_lambda_, p, n, cl, max_iter = max_iter, lr = lr, momentum = momentum, alpha = alpha, eps = eps, centered = centered, w = w, early_stopping = early_stopping, verbose = verbose)
    
    elif method in ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']:
        # Minimization by scipy.optimize.minimize
        options = {'maxiter':max_iter, 'disp':verbose}
        tau = tau_init
        if verbose:
            print("First loss :", cl.loss(tau, train_lambda_, p, n))
        res = minimize(cl.loss, x0 = tau_init, args = (train_lambda_, p, n), jac = cl.jacobian, hess = cl.hessian, method = method, options=options)
        tau = np.sort(res.x)
        if verbose:
            print(res.message)
            print("Final loss epoch :", cl.loss(tau, train_lambda_, p, n))
    return tau


# Non-linear shrinkage

def nonlinear_shrinkage(tau, lambda_sample, p, n):
    """
        Inputs:
            tau: array of shape (p), population eigenvalues
            lambda_sample: float array of shape (n, p), sample eigenvalues
            p: int, the dimension
            n: int, the number of samples
        Outputs:
            lambda_shrunk: float array of shape (p), non linearly shrunk sample eigenvalues
    """    
    c = p/n
    
    # Sort
    tau = np.maximum(tau,0)
    tau = np.sort(tau)
    
    # Partially compute the QuEST function
    nu, u_nu, omega_nu, w, t = support(tau, c)
    xi = grid(nu, u_nu, omega_nu)
    y = find_y(w, t, tau, nu, xi, c)
    z = compute_z(y, xi, nu)
    x, f = density(z, tau, nu, c)
    
    # Make z and x as arrays
    x_flat = np.concatenate([np.concatenate(x, axis=0), np.inf*np.ones(1)], axis=0)
    z_flat = np.concatenate([np.concatenate(z, axis=0), z[-1][-1]*np.ones(1)], axis=0)
    
    # Linear interpolation between z[j] and z[j+1], with x[j] < lambda < x[j+1]
    idx = (lambda_sample[:,np.newaxis] >= x_flat[np.newaxis,:-1]) & (lambda_sample[:,np.newaxis] < x_flat[np.newaxis,1:])
    j_idx = np.argmax(idx, axis=1).astype(int)
    v = z_flat[j_idx] + (lambda_sample - x_flat[j_idx])/(x_flat[j_idx+1] - x_flat[j_idx])*(z_flat[j_idx+1] - z_flat[j_idx])
    m_F = (1 - c)/(c*lambda_sample) - 1/c/v
    
    # Non linear shrinkage
    lambda_shrunk = lambda_sample/np.abs(1 - c - c*lambda_sample*m_F)**2
    
    return np.sort(lambda_shrunk)

def NL_covariance(tau, X_sample, assume_centered = False):
    """
        Inputs:
            tau: array of shape (p), population eigenvalues
            X_sample: float array of sahpe (n, p), with n the number of samples, and p the dimension
            assume_centered: bool, if False, the data will be demeaned
        Outputs:
            shrunk_covariance: float array of shape (p, p), non linearly shrunk sample covariance
    """    
    n, p = X_sample.shape
    
    if assume_centered:
        sample_cov = X_sample.T @ X_sample/n
    else:
        X_mean = X_sample.mean(axis=0)[np.newaxis,:]
        sample_cov = (X_sample - X_mean).T @ (X_sample - X_mean)/(n-1)
        
    lambda_sample, V = sp.linalg.eigh(sample_cov)
    shrunk_eigenvalues = nonlinear_shrinkage(tau, lambda_sample, p, n)
    shrunk_covariance = V @ np.diag(shrunk_eigenvalues) @ V.T
    
    return shrunk_covariance

if __name__ == '__main__':
    import time
    from sklearn.covariance import LedoitWolf
    
    
    n = 2000
    p = 200
    tau_pop = np.concatenate([np.linspace(1.9, 2.1, p//2), np.linspace(4.9, 5.1, p - p//2)])
    
    # Jacobian verifications
    jacobian_check = False
    if jacobian_check:
        jac_u_a, jac_xi_a, jac_y_a, jac_mlh_a, jac_x_a, jac_f_a, jac_F_a, jac_X_a, jac_tau_a, jac_u, jac_xi, jac_y, jac_mlh, jac_x, jac_f, jac_F, jac_X, jac_tau = jacobian(tau_pop, p, n, eps = 1e-8)
        
        print(jac_tau_a)
        print(jac_tau)
        print(np.linalg.norm(jac_tau_a - jac_tau, ord='fro')/np.linalg.norm(jac_tau, ord='fro'))
    
    # Minimization
    minim = False
    if minim:
        # Params
        max_iter = 50
        lr = 5e-2
        dist = 'l2'
        method = 'Adam'
        regu = 1
        power = 1
        # in ['GD', 'Adam', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
        # Exp:
            # GD: lr = 5e-2, experimentally most robust and a.s converges
            # Adam: lr = 1e-2, experimentally fastest at the begin but stops converging quickly, used by default
            # Composite: lr = 5e-2, mix of Adam then GD
            # RMSProp: subject to divergence
            # Nelder-Mead: does not converge, very slow
            # Powell: exits with an error
            # CG: does not converge due precision loss
            # BFGS: does not converge due precision loss
            # TNC: very slow, does not converge
            # SLSQP: exits with an error
            # Newton-CG: CG iterations didn't converge. The Hessian is not positive definite.
            # dogleg: A linalg error occurred, such as a non-psd Hessian.
            # trust-ncg: A bad approximation caused failure to predict improvement.
            # trust-krylov: A bad approximation caused failure to predict improvement.
            # trust-exact: A bad approximation caused failure to predict improvement.
            
        tol = 1e-3
        betas = (0.9,0.999)
        eps = 1e-8
        alpha = 0.9
        momentum = 0.
        w = 0.
        centered = False
        early_stopping = True
        
        # Sampling
        cov = np.diag(tau_pop)
        X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))
        
        # Initialization
        LW_cov = LedoitWolf().fit(X_sample).covariance_
        tau_init, _ = sp.linalg.eigh(LW_cov)
        
        
        # Minimization
        start_time = time.time()
        res = QuEST_minimization(X_sample, tau_init, dist = dist, power= power, regu = regu, method = method, assume_centered = True, tol = tol, max_iter = max_iter, lr = lr, betas = betas, eps = eps, w = w)
        tau_quest = np.sort(res)
        last_time = time.time()
        print("Compute time:", last_time - start_time, "s.")
        
        # Non-linear shrinkage
        sample_cov = X_sample.T @ X_sample/n
        shrunk_covariance = NL_covariance(tau_quest, X_sample, assume_centered = True)
        ex_shrunk_covariance = NL_covariance(tau_pop, X_sample, assume_centered = True)
        print("Lin covariance PRIAL:\t", 1 - np.linalg.norm(LW_cov - cov)**2/np.linalg.norm(sample_cov - cov)**2)
        print("NL covariance PRIAL:\t", 1 - np.linalg.norm(shrunk_covariance - cov)**2/np.linalg.norm(sample_cov - cov)**2)
        print("Ex. NL covariance PRIAL:", 1 - np.linalg.norm(ex_shrunk_covariance - cov)**2/np.linalg.norm(sample_cov - cov)**2)
        
        # Plotting
        plt.figure()
        plt.plot(tau_pop, label="population eigenvalues")
        plt.plot(tau_init, label="inital eigenvalues")
        plt.plot(tau_quest, label='QuEST estimated eigenvalues')
        plt.title("QuEST inversion, method "+method)
        plt.legend()
        plt.show()
        
        
        # Initial plot
        lambda_unsrt1, dlambdadtau_unsrt1, x1, f1, x_tilde1, F1 =  QuEST(tau_pop, p, n)
        lambda_unsrt2, dlambdadtau_unsrt2, x2, f2, x_tilde2, F2 =  QuEST(tau_init, p, n)
        lambda_unsrt3, dlambdadtau_unsrt3, x3, f3, x_tilde3, F3 =  QuEST(tau_quest, p, n)
        
        plt.figure()
        plt.plot(x1[0], f1[0], label=r'PDF from pop')
        plt.plot(x2[0], f2[0], label=r'PDF from init')
        plt.plot(x3[0], f3[0], label=r'PDF from QuEST inv')
        plt.title("Sample PDF, method "+method)
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(x_tilde1[1], F1[1], label=r'CDF from pop')
        plt.plot(x_tilde2[1], F2[1], label=r'CDF from init')
        plt.plot(x_tilde3[1], F3[1], label=r'CDF from QuEST inv')
        plt.title("Sample CDF, method "+method)
        plt.legend()
        plt.show()























