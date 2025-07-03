from scipy.optimize import minimize, LinearConstraint
import numpy as np
import scipy as sp
from scipy.special import gamma
from LWO_estimator import LWO_estimator
        
class CCC:
    """
        See https://www.diva-portal.org/smash/get/diva2:1324825/FULLTEXT01.pdf
    """
    def __init__(self, p1 = 1, q1 = 1, dist = 'norm'):
        if dist == 'norm' or dist == 't' or dist == 'skewt':
            self.dist = dist
        else: 
            print("Takes pdf name as param: 'norm' or 't'.")
        self.p1 = p1
        self.q1 = q1
            
    def garch_fit(self, returns):
        if self.dist == 'norm':
            res = minimize(self.garch_loglike, [returns.var()]+[0.01/self.p1]*self.p1+[0.94/self.q1]*self.q1, args = returns,
                  bounds = [(1e-6*returns.var(), 10*returns.var())] + [(1e-6, 1)]*(self.p1 + self.q1))
        elif self.dist == 't':
            res = minimize(self.garch_loglike, [returns.var()]+[0.01/self.p1]*self.p1+[0.94/self.q1]*self.q1+[5], args = returns,
                  bounds = [(1e-6*returns.var(), 10*returns.var())] + [(1e-6, 1)]*(self.p1 + self.q1))
        else:
            res = minimize(self.garch_loglike, [returns.var()]+[0.01/self.p1]*self.p1+[0.94/self.q1]*self.q1, args = returns,
                  bounds = [(1e-6*returns.var(), 10*returns.var())] + [(1e-6, 1)]*(self.p1 + self.q1))
        return res.x

    def garch_loglike(self, params, returns):
        var_t = self.garch_var(params, returns)
        if self.dist == "norm":
            LogL = -np.log(2*np.pi*var_t**2).sum() - (returns**2/var_t).sum()
        elif self.dist == "t":
            dof = params[-1]
            LogL = - np.log(var_t).sum()/2. 
            LogL += - (dof+ 1)/2.*np.log(1 + returns**2/var_t/(dof - 2.)).sum()
            LogL += self.T*np.log(gamma((1+dof)/2)) 
            LogL += - self.T*np.log(gamma(dof/2))
            LogL += - self.T*(1/2.)*np.log(np.pi*(dof - 2))
        else:
            LogL = -np.log(2*np.pi*var_t**2).sum() - (returns**2/var_t).sum()
        return -LogL

    def garch_var(self, params, returns, var_0 = None):
        omega = params[0]
        alpha = np.array(params[1:1+self.p1])
        beta = np.array(params[1+self.p1:1+self.p1+self.q1])
        var_t = np.zeros(self.T)   
        
        try:
            if var_0 == None:
                var_t[0] = omega
            elif var_0 != None:
                var_t[0] = var_0
        except:
            var_t[0] = omega
        
        for i in range(1,self.T):
            var_t[i] = (1 + alpha[:-(i-max(i-self.p1,0))].sum() + beta[:-(i-max(i-self.q1,0))].sum())*omega + (alpha[-(i-max(i-self.p1,0)):]*(returns[max(i-self.p1,0):i]**2)).sum(axis=0) + (beta[-(i-max(i-self.q1,0)):]*var_t[max(i-self.q1,0):i]).sum(axis=0)

        return var_t        
    
    def CCC_loglike(self, complete=False):
        self.H_t = H_t = self.D_t[:,:,np.newaxis] * self.Q_bar[np.newaxis,:,:] * self.D_t[:,np.newaxis,:]
        
        loglike = 0
        loglike += self.T*np.linalg.slogdet(self.Q_bar)[1]/2 
        loglike += (self.et[:,np.newaxis,:] @ np.linalg.inv(self.Q_bar)[np.newaxis,:,:] @ self.et[:,:,np.newaxis]/2).sum()
            
        if complete:
            loglike += self.N*self.T/2*np.log(2*np.pi) + np.log(self.D_t).sum()
        
        return loglike/self.T
    
    def CCC_logliket(self, params, complete = False):
        # No of assets
        dof = params[0] 

        loglike = 0
        loglike += self.T*np.log(gamma((self.N+dof)/2)) 
        loglike += - self.T*np.log(gamma(dof/2))
        loglike += - self.T*(self.N/2.)*np.log(np.pi*(dof - 2))
        loglike += - self.T*np.linalg.slogdet(self.Q_bar)[1]/2. 
        
        Q_inv = np.linalg.inv(self.Q_bar)
        for i in range(self.T):
            loglike += - (dof+ self.N)/2.*np.log(1 + self.et[i,np.newaxis,:] @ Q_inv @ self.et[i,:,np.newaxis]/(dof - 2.))

        if complete:
            loglike += - np.log(self.D_t).sum()

        return -loglike[0,0]/self.T
    
    def CCC_loglikeskewt(self, params, complete = False):
        # No of assets
        dof = params[0]
        c = np.array(params[1:])[:,np.newaxis] # shape (N,1)
        
        def T1(x, nu):
            return 1/(nu*np.pi)*gamma((nu+1)/2)/gamma(nu/2)*(1 + x**2/nu)**(-(nu+1)/2)

        loglike = 0        
        K = np.sqrt(1 + 4*dof*(dof-2)*(np.pi*gamma(dof/2)**2 - (dof-2)*gamma((dof-1)/2)**2)*c.T @ c/(np.pi*gamma(dof/2)**2*(dof - (dof-2)*c.T @ c)**2))
        if (c**2).sum() != 0:
            omega = (dof-2)/dof*(np.eye(self.N) + 1/(c.T @ c)*(-1 + np.pi*gamma(dof/2)**2*(dof - (dof-2)*c.T @ c)/(2*c.T @ c*(dof-2)*(np.pi*gamma(dof/2)**2 - (dof-2)*gamma((dof-1)/2)**2))*(K - 1))* c @ c.T)
        else:
            omega = (dof-2)/dof*np.eye(self.N)
        D = np.diag(np.sqrt(np.diag(omega))) # shape (N,N)
        D_inv = np.diag(np.sqrt(1/np.diag(omega)))
        delta = D @ c # shape (N,1)
        om_inv = np.linalg.inv(omega)
        for i in range(self.T):
            xi = - np.sqrt(dof/np.pi) * gamma((dof-1)/2) / gamma(dof/2)/np.sqrt(1 + c.T @ omega @ c) * omega @ c # shape (N,1)
            Qat = (self.et[i,np.newaxis,:] - xi.T) @ om_inv @ (self.et[i,:,np.newaxis] - xi)
            loglike += np.log(T1(delta.T @ D_inv @ (self.et[i,:,np.newaxis] - xi)*np.sqrt((dof+self.N)/(Qat+dof)), dof+self.N))
            loglike += - (dof+ self.N)/2.*np.log(1 + Qat/dof)
            
        loglike += self.T*np.log(gamma((self.N+dof)/2))  
        loglike += - self.T*np.linalg.slogdet(omega)[1]/2 
        loglike += - self.T*self.N/2*np.log(np.pi*dof)
        loglike += - self.T*np.log(gamma(dof/2))
        loglike += - self.T*np.linalg.slogdet(self.Q_bar)[1]/2. 

        if complete:
            loglike +=0 # not implemented

        return -loglike[0,0]/self.T
    
    def loglike(self, returns_test, complete = True, continuity = True, method="default"):
        omega = self.garch_params[:,0]
        alpha = np.array(self.garch_params[:,1:1+self.p1])
        beta = np.array(self.garch_params[:,1+self.p1:])
        if continuity:
            var_0 = omega + (alpha*self.rt[-self.p1:,:].T**2).sum(axis=1) + (beta*self.D_t[-self.q1:,:].T).sum(axis=1)
        else:
            var_0 = None
        
        self.rt = returns_test
        self.T = self.rt.shape[0]
        
        # Demeaning
        self.rt = self.rt - self.mean
        
        # GARCH(1,1)
        D_t = np.zeros((self.T, self.N))
        for i in range(self.N):
            D_t[:,i] = np.sqrt(self.garch_var(self.garch_params[i], self.rt[:,i], var_0=var_0[i]))
        self.D_t = D_t
        
        # Devolatilized returns
        self.et = self.rt/D_t
            
        if self.dist == "norm":
            llk = self.CCC_loglike(complete=complete)
                
        if self.dist == "t":
            params = [self.dof]
            llk = self.CCC_logliket(params, complete=complete)
        
        if self.dist == "skewt":
            params = [self.dof] + list(self.c)
            llk = self.CCC_loglikeskewt(params, complete=complete)
        
        return llk
    
    def predict(self, seq_length, continuity_index = None):
        """
            continuity_index:   if None, intialization with omega and Q_bar
                                if int, will use et[continuity_index], r_t[continuity_index] and D_t[continuity_index] to initialize

        """
        gen_D_t = np.zeros((seq_length, self.N))    
        gen_H_t = np.zeros((seq_length,self.N,self.N))
        gen_rt = np.zeros((seq_length, self.N))
        
        omega = self.garch_params[:,0]
        alpha = self.garch_params[:,1:1+self.p1]
        beta = self.garch_params[:,1+self.p1:1+self.p1+self.q1]             
        
        # Initialization
        if continuity_index == None:
            gen_D_t[0,:] = np.sqrt(omega)
            gen_H_t[0,:] = np.diag(gen_D_t[0,:]) @ self.Q_bar @ np.diag(gen_D_t[0,:])
        else:
            i = continuity_index
            gen_D_t[0,:] = np.sqrt((1 + alpha[:,:-(i-max(i-self.p1,0))].sum(axis=1) + beta[:,:-(i-max(i-self.q1,0))].sum(axis=1))*omega + (alpha[:,-(i-max(i-self.p1,0)):].T*(self.rt[max(i-self.p1,0):i]**2)).sum(axis=0) + (beta[:,-(i-max(i-self.q1,0)):].T*self.D_t[max(i-self.q1,0):i]**2).sum(axis=0))
            gen_H_t[0,:] = np.diag(gen_D_t[0,:]) @ self.Q_bar @ np.diag(gen_D_t[0,:])
        
        if self.dist == "norm":
            gen_et = np.random.normal(size = (seq_length, self.N))
        if self.dist == "t":
            gen_et = sp.stats.multivariate_t(df = self.dof, allow_singular = True).rvs(size = (seq_length, self.N))
        if self.dist == "skewt":
            # need to implement a sampler following the fomulation of https://arxiv.org/pdf/0911.2342.pdf
            # in the mean time, only Student is supported
            raise NotImplementedError("Skew-t model non supported for prediction")
            
        gen_rt[0,:] = sp.linalg.sqrtm(gen_H_t[0,:]) @ gen_et[0,:]
        
        # Iterative generation
        for i in range(1, seq_length):
            gen_D_t[i,:] = np.sqrt((1 + alpha[:,:-(i-max(i-self.p1,0))].sum(axis=1) + beta[:,:-(i-max(i-self.q1,0))].sum(axis=1))*omega + (alpha[:,-(i-max(i-self.p1,0)):].T*(gen_rt[max(i-self.p1,0):i]**2)).sum(axis=0) + (beta[:,-(i-max(i-self.q1,0)):].T*gen_D_t[max(i-self.q1,0):i]**2).sum(axis=0))
            gen_H_t[i,:] = np.diag(gen_D_t[i,:]) @ self.Q_bar @ np.diag(gen_D_t[i,:])
            gen_rt[i,:] = sp.linalg.sqrtm(gen_H_t[i,:]) @ gen_et[i,:]
        
        return gen_rt + self.mean[np.newaxis,:], gen_H_t
            
            
    def fit(self, returns, verbose = True):
        if verbose:
            print("Calibration of CCC, univariate GARCH("+str(self.p1)+","+str(self.q1)+"), distrib "+self.dist)
        
        self.rt = returns
        self.T = self.rt.shape[0]
        self.N = self.rt.shape[1]
        
        if self.N == 1 or self.T == 1:
            return 'Required: 2d-array with columns > 2' 
        
        # Demeaning
        self.mean = self.rt.mean(axis = 0)
        self.rt = self.rt - self.mean
        
        # GARCH(1,1) calibration
        D_t = np.zeros((self.T, self.N))
        garch_params = np.zeros((self.N, 1+self.p1+self.q1))
        for i in range(self.N):
            if verbose:
                print("\rGARCH calibration of asset "+str(i+1)+"/"+str(self.N), end="")
            garch_params[i] = self.garch_fit(self.rt[:,i])[:1+self.p1+self.q1]
            D_t[:,i] = np.sqrt(self.garch_var(garch_params[i], self.rt[:,i]))
        if verbose:
            print()
        self.garch_params = garch_params
        self.D_t = D_t
        
        # Devolatilized returns
        self.et = self.rt/D_t
        
        # Targeted correlation and normalization 
        Q_bar = LWO_estimator(self.et, assume_centered = False)
        #Q_bar = self.et.T @ self.et/(self.T-1)
        Q_bar_sqinv = np.sqrt(np.diag(1/np.diag(Q_bar)))
        Q_bar = Q_bar_sqinv @ Q_bar @ Q_bar_sqinv
        self.Q_bar = Q_bar
                
        if self.dist == 'norm':            
            if verbose:
                last_likelihood = self.CCC_loglike(complete=True)
                print("Last neg-llikelihood:\t", last_likelihood)
                
            return {'mu': self.mean} 
        elif self.dist == 't':
            start_params = [5.]
            first_likelihood = self.CCC_logliket(start_params, complete=True)
            
            if verbose:
                print("First neg-llikelihood:\t", first_likelihood)
                
            res = minimize(self.CCC_logliket, start_params,
            bounds = [(3., None)], tol=1e-6, method='SLSQP',
            options = {'maxiter':10000000, 'disp':True},
            )
            self.dof = res.x[0]
            
            if verbose:
                last_likelihood = self.CCC_logliket(res.x, complete=True)
                print("Last neg-llikelihood:\t", last_likelihood)
                
            return {'mu': self.mean, 'dof': self.dof} 
        
        elif self.dist == 'skewt':
            start_params = [10]+ list(np.random.normal(size=self.N)/100)
            first_likelihood = self.CCC_loglikeskewt(start_params, complete=True)
            
            if verbose:
                print("First neg-llikelihood:\t", first_likelihood)
            
            res = minimize(self.CCC_loglikeskewt, start_params,
            bounds = [(3, None)] + [(None, None)]*self.N, tol=1e-6, method='SLSQP',
            options = {'maxiter':10000000, 'disp':True},
            )
            self.dof = res.x[0]
            self.c = np.array(res.x[1:])
            
            if verbose:
                last_likelihood = self.CCC_loglikeskewt(res.x, complete=True)
                print("Last neg-llikelihood:\t", last_likelihood)
            return {'mu': self.mean, 'dof': self.dof, 'c': self.c} 
        else:
            raise NotImplementedError("Model non supported for fitting")