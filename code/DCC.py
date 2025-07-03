from scipy.optimize import minimize, LinearConstraint
import numpy as np
import scipy as sp
from scipy.special import gamma
from LWO_estimator import LWO_estimator
        
class DCC:
    """
        See https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/259296/724505_FULLTEXT01.pdf
        https://core.ac.uk/download/pdf/235049858.pdf suggests that (1,1) may be the better choice
    """
    def __init__(self, p1 = 1, q1 = 1, p2 = 1, q2 = 1, dist = 'norm'):
        if dist == 'norm' or dist == 't' or dist == 'skewt':
            self.dist = dist
        else: 
            print("Takes pdf name as param: 'norm' or 't' or 'skewt.")
        self.p1 = p1
        self.q1 = q1
        self.p2 = p2
        self.q2 = q2
            
    def garch_fit(self, returns):
        if self.dist == 'norm':
            res = minimize(self.garch_loglike, [returns.var()]+[0.01/self.p1]*self.p1+[0.94/self.q1]*self.q1, args = returns,
                  bounds = [(1e-6*returns.var(), 10*returns.var())] + [(1e-6, 1)]*(self.p1 + self.q1))
        elif self.dist == 't':
            res = minimize(self.garch_loglike, [returns.var()]+[0.01/self.p1]*self.p1+[0.94/self.q1]*self.q1+[5], args = returns,
                  bounds = [(1e-6*returns.var(), 10*returns.var())] + [(1e-6, 1)]*(self.p1 + self.q1)+[(3,None)])
        else:
            res = minimize(self.garch_loglike, [returns.var()]+[0.01/self.p1]*self.p1+[0.94/self.q1]*self.q1, args = returns,
                  bounds = [(1e-6*returns.var(), 10*returns.var())] + [(1e-6, 1)]*(self.p1 + self.q1))
        return res.x

    def garch_loglike(self, params, returns):
        var_t = self.garch_var(params, returns)
        if self.dist == "norm":
            LogL = - np.log(2*np.pi*var_t**2).sum() - (returns**2/var_t).sum()
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
        
    def DCC_loglike(self, params, complete=False):
        a = np.array(params[:self.p2])
        b = np.array(params[self.p2:self.p2+self.q2])

        Q_t = np.zeros((self.T,self.N,self.N))
        R_t = np.zeros((self.T,self.N,self.N))
        
        Q_t[0] = self.Q_bar

        loglike = 0
        for i in range(1,self.T):
            Q_t[i] = (1-a[:-(i-max(i-self.p2,0))].sum()-b[:-(i-max(i-self.q2,0))].sum())*self.Q_bar + (a[-(i-max(i-self.p2,0)):, None, None]*(self.et[max(i-self.p2,0):i,:,np.newaxis]*self.et[max(i-self.p2,0):i,np.newaxis,:])).sum(axis=0) + (b[-(i-max(i-self.q2,0)):, None, None]*Q_t[max(i-self.q2,0):i]).sum(axis=0)
            qts = np.sqrt(np.diag(1/np.diag(Q_t[i])))

            R_t[i] = qts @ Q_t[i] @ qts
            Q_t[i] = R_t[i]
            
            loglike += np.linalg.slogdet(R_t[i])[1]/2 
            loglike += self.et[i,np.newaxis,:] @ np.linalg.inv(R_t[i]) @ self.et[i,:,np.newaxis]/2
        
        if complete:
            loglike += self.T*self.N/2*np.log(2*np.pi) + np.log(self.D_t).sum()
        
        return loglike[0,0]/self.T
    
    def DCC_loglike_2MSCLE(self, params, complete=False):
        """
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1354497
        """
        a = np.array(params[:self.p2])
        b = np.array(params[self.p2:self.p2+self.q2])
        
        Q_t = np.zeros((self.T,self.N,self.N))
        R_t = np.zeros((self.T,self.N,self.N))
        
        Q_t[0] = self.Q_bar
        
        idx = np.ones((2,self.N-1,2,2)).cumsum(axis=1) - 1
        idx[:,:,1,1] += 1
        idx[1,:,0,1] += 1
        idx[0,:,1,0] += 1
        idx = tuple(idx.astype(int))
        
        loglike = 0
        for i in range(1,self.T):
            Q_t[i] = (1-a[:-(i-max(i-self.p2,0))].sum()-b[:-(i-max(i-self.q2,0))].sum())*self.Q_bar + (a[-(i-max(i-self.p2,0)):, None, None]*(self.et[max(i-self.p2,0):i,:,np.newaxis]*self.et[max(i-self.p2,0):i,np.newaxis,:])).sum(axis=0) + (b[-(i-max(i-self.q2,0)):, None, None]*Q_t[max(i-self.q2,0):i]).sum(axis=0)
            qts = np.sqrt(np.diag(1/np.diag(Q_t[i])))
        
            R_t[i] = qts @ Q_t[i] @ qts
            Q_t[i] = R_t[i]
            
            R_2 = R_t[i][idx] #shape (self.N-1,2,2)
            et2 = self.et[i][idx[0][:,:,0]]
            
            loglike += np.log(1 - R_2[:,0,1]**2).sum()/2 
            loglike += (et2[:,np.newaxis,:] @ np.linalg.inv(R_2) @ et2[:,:,np.newaxis]).sum()/2
            
        if complete:
            loglike += self.T*self.N/2*np.log(2*np.pi) + np.log(self.D_t).sum()
        
        return loglike/self.T
         
    
    def DCC_logliket(self, params, complete = False):
        a = np.array(params[:self.p2])
        b = np.array(params[self.p2:self.p2+self.q2])
        dof = params[-1]

        Q_t = np.zeros((self.T,self.N,self.N))
        R_t = np.zeros((self.T,self.N,self.N))
        
        Q_t[0] = self.Q_bar 

        loglike = 0
        for i in range(1,self.T):
            Q_t[i] = (1-a[:-(i-max(i-self.p2,0))].sum()-b[:-(i-max(i-self.q2,0))].sum())*self.Q_bar + (a[-(i-max(i-self.p2,0)):, None, None]*(self.et[max(i-self.p2,0):i,:,np.newaxis]*self.et[max(i-self.p2,0):i,np.newaxis,:])).sum(axis=0) + (b[-(i-max(i-self.q2,0)):, None, None]*Q_t[max(i-self.q2,0):i]).sum(axis=0)
            qts = np.sqrt(np.diag(1/np.diag(Q_t[i])))

            R_t[i] = qts @ Q_t[i] @ qts
            Q_t[i] = R_t[i]

            loglike += - np.linalg.slogdet(R_t[i])[1]/2. 
            loglike += - (dof+ self.N)/2.*np.log(1 + self.et[i,np.newaxis,:] @ np.linalg.inv(R_t[i]) @ self.et[i,:,np.newaxis]/(dof - 2.))
            
        loglike += self.T*np.log(gamma((self.N+dof)/2)) 
        loglike += - self.T*np.log(gamma(dof/2))
        loglike += - self.T*(self.N/2.)*np.log(np.pi*(dof - 2))
        if complete:
            loglike += - np.log(self.D_t).sum()

        return -loglike[0,0]/self.T
    
    def DCC_logliket_2MSCLE(self, params, complete = False):
        a = np.array(params[:self.p2])
        b = np.array(params[self.p2:self.p2+self.q2])
        dof = params[-1]

        Q_t = np.zeros((self.T,self.N,self.N))
        R_t = np.zeros((self.T,self.N,self.N))
        
        Q_t[0] = self.Q_bar
        
        idx = np.ones((2,self.N-1,2,2)).cumsum(axis=1) - 1
        idx[:,:,1,1] += 1
        idx[1,:,0,1] += 1
        idx[0,:,1,0] += 1
        idx = tuple(idx.astype(int))
        
        loglike = 0
        for i in range(1,self.T):
            Q_t[i] = (1-a[:-(i-max(i-self.p2,0))].sum()-b[:-(i-max(i-self.q2,0))].sum())*self.Q_bar + (a[-(i-max(i-self.p2,0)):, None, None]*(self.et[max(i-self.p2,0):i,:,np.newaxis]*self.et[max(i-self.p2,0):i,np.newaxis,:])).sum(axis=0) + (b[-(i-max(i-self.q2,0)):, None, None]*Q_t[max(i-self.q2,0):i]).sum(axis=0)
            qts = np.sqrt(np.diag(1/np.diag(Q_t[i])))

            R_t[i] = qts @ Q_t[i] @ qts
            Q_t[i] = R_t[i]
            
            R_2 = R_t[i][idx] #shape (self.N-1,2,2)
            et2 = self.et[i][idx[0][:,:,0]]
                        
            loglike += - np.log(1 - R_2[:,0,1]**2).sum()/2. 
            loglike += - (dof+ self.N)/2.*np.log(1 + et2[:,np.newaxis,:] @ np.linalg.inv(R_2) @ et2[:,:,np.newaxis]/(dof - 2.)).sum()

        loglike += self.T*np.log(gamma((self.N+dof)/2)) 
        loglike += - self.T*np.log(gamma(dof/2))
        loglike += - self.T*(self.N/2.)*np.log(np.pi*(dof - 2))
        if complete:
            loglike += - np.log(self.D_t).sum()

        return -loglike/self.T
    
    def DCC_loglikeskewt(self, params, complete = False):
        a = np.array(params[:self.p2])
        b = np.array(params[self.p2:self.p2+self.q2])
        dof = params[self.p2+self.q2]
        c = np.array(params[self.p2+self.q2+1:])[:,np.newaxis] # shape (N,1)

        Q_t = np.zeros((self.T,self.N,self.N))
        R_t = np.zeros((self.T,self.N,self.N))
        
        Q_t[0] = self.Q_bar 
        
        def T1(x, nu):
            return 1/(nu*np.pi)*gamma((nu+1)/2)/gamma(nu/2)*(1 + x**2/nu)**(-(nu+1)/2)

        K = np.sqrt(1 + 4*dof*(dof-2)*(np.pi*gamma(dof/2)**2 - (dof-2)*gamma((dof-1)/2)**2)*c.T @ c/(np.pi*gamma(dof/2)**2*(dof - (dof-2)*c.T @ c)**2))
        if (c**2).sum() != 0:
            omega = (dof-2)/dof*(np.eye(self.N) + 1/(c.T @ c)*(-1 + np.pi*gamma(dof/2)**2*(dof - (dof-2)*c.T @ c)/(2*c.T @ c*(dof-2)*(np.pi*gamma(dof/2)**2 - (dof-2)*gamma((dof-1)/2)**2))*(K - 1))* c @ c.T)
        else:
            omega = (dof-2)/dof*np.eye(self.N)
        D = np.diag(np.sqrt(np.diag(omega))) # shape (N,N)
        D_inv = np.diag(np.sqrt(1/np.diag(omega)))
        delta = D @ c # shape (N,1)
        om_inv = np.linalg.pinv(omega)
        xi = - np.sqrt(dof/np.pi) * gamma((dof-1)/2) / gamma(dof/2)/np.sqrt(1 + c.T @ omega @ c) * omega @ c # shape (N,1)
        
        loglike = 0
        for i in range(1,self.T):
            Q_t[i] = (1-a[:-(i-max(i-self.p2,0))].sum()-b[:-(i-max(i-self.q2,0))].sum())*self.Q_bar + (a[-(i-max(i-self.p2,0)):, None, None]*(self.et[max(i-self.p2,0):i,:,np.newaxis]*self.et[max(i-self.p2,0):i,np.newaxis,:])).sum(axis=0) + (b[-(i-max(i-self.q2,0)):, None, None]*Q_t[max(i-self.q2,0):i]).sum(axis=0)
            qts = np.sqrt(np.diag(1/np.diag(Q_t[i])))

            R_t[i] = qts @ Q_t[i] @ qts
            Q_t[i] = R_t[i]
            
            Qat = (self.et[i,np.newaxis,:] - xi.T) @ om_inv @ (self.et[i,:,np.newaxis] - xi)
            
            loglike += np.log(gamma((self.N+dof)/2))  
            loglike += - np.linalg.slogdet(omega)[1]/2 
            loglike += - self.N/2*np.log(np.pi*dof)
            loglike += - np.log(gamma(dof/2))
            loglike += - (dof+ self.N)/2.*np.log(1 + Qat/dof)
            loglike += np.log(T1(delta.T @ D_inv @ (self.et[i,:,np.newaxis] - xi)*np.sqrt((dof+self.N)/(Qat+dof)), dof+self.N))
            loglike += - np.linalg.slogdet(R_t[i])[1]/2. 

            if complete:
                loglike +=0 # not implemented

        return -loglike[0,0]/self.T
    
    def DCC_loglikeskewt_2MSCLE(self, params, complete = False):
        a = np.array(params[:self.p2])
        b = np.array(params[self.p2:self.p2+self.q2])
        dof = params[self.p2+self.q2]
        c = np.array(params[self.p2+self.q2+1:])[:,np.newaxis] # shape (N,1)

        Q_t = np.zeros((self.T,self.N,self.N))
        R_t = np.zeros((self.T,self.N,self.N))
        
        Q_t[0] = self.Q_bar 
        
        idx = np.ones((2,self.N-1,2,2)).cumsum(axis=1) - 1
        idx[:,:,1,1] += 1
        idx[1,:,0,1] += 1
        idx[0,:,1,0] += 1
        idx = tuple(idx.astype(int))
        
        def T1(x, nu):
            return 1/(nu*np.pi)*gamma((nu+1)/2)/gamma(nu/2)*(1 + x**2/nu)**(-(nu+1)/2)

        K = np.sqrt(1 + 4*dof*(dof-2)*(np.pi*gamma(dof/2)**2 - (dof-2)*gamma((dof-1)/2)**2)*c.T @ c/(np.pi*gamma(dof/2)**2*(dof - (dof-2)*c.T @ c)**2))
        if (c**2).sum() != 0:
            omega = (dof-2)/dof*(np.eye(self.N) + 1/(c.T @ c)*(-1 + np.pi*gamma(dof/2)**2*(dof - (dof-2)*c.T @ c)/(2*c.T @ c*(dof-2)*(np.pi*gamma(dof/2)**2 - (dof-2)*gamma((dof-1)/2)**2))*(K - 1))* c @ c.T)
        else:
            omega = (dof-2)/dof*np.eye(self.N)
        D = np.diag(np.sqrt(np.diag(omega))) # shape (N,N)
        D_inv = np.diag(np.sqrt(1/np.diag(omega)))
        D_inv2 = D_inv[idx]
        delta = D @ c # shape (N,1)
        delta2 = delta[:,0][idx[0][:,:,0]] # shape (N-1,2)s
        om_inv = np.linalg.pinv(omega)
        om_inv2 = om_inv[idx]
        xi = - np.sqrt(dof/np.pi) * gamma((dof-1)/2) / gamma(dof/2)/np.sqrt(1 + c.T @ omega @ c) * omega @ c # shape (N,1)
        xi2 = xi[:,0][idx[0][:,:,0]] # shape (N-1,2)
        
        loglike = 0
        for i in range(1,self.T):
            Q_t[i] = (1-a[:-(i-max(i-self.p2,0))].sum()-b[:-(i-max(i-self.q2,0))].sum())*self.Q_bar + (a[-(i-max(i-self.p2,0)):, None, None]*(self.et[max(i-self.p2,0):i,:,np.newaxis]*self.et[max(i-self.p2,0):i,np.newaxis,:])).sum(axis=0) + (b[-(i-max(i-self.q2,0)):, None, None]*Q_t[max(i-self.q2,0):i]).sum(axis=0)
            qts = np.sqrt(np.diag(1/np.diag(Q_t[i])))

            R_t[i] = qts @ Q_t[i] @ qts
            Q_t[i] = R_t[i]
            
            R_2 = R_t[i][idx] #shape (self.N-1,2,2)
            et2 = self.et[i][idx[0][:,:,0]]
            
            Qat = (et2[:,np.newaxis,:] - xi2[:,np.newaxis,:]) @ om_inv2 @ (et2[:,:,np.newaxis] - xi2[:,:,np.newaxis])
            
            loglike += - (dof+ self.N)/2.*np.log(1 + Qat/dof).sum()
            loglike += np.log(T1(delta2[:,np.newaxis,:] @ D_inv2 @ (et2[:,:,np.newaxis] - xi2[:,:,np.newaxis])*np.sqrt((dof+self.N)/(Qat+dof)), dof+self.N)).sum()
            loglike += - np.log(1 - R_2[:,0,1]**2).sum()/2. 

        loglike += self.T*np.log(gamma((self.N+dof)/2))  
        loglike += - self.T*np.linalg.slogdet(omega)[1]/2 
        loglike += - self.T*self.N/2*np.log(np.pi*dof)
        loglike += - self.T*np.log(gamma(dof/2))
        
        if complete:
            loglike +=0 # not implemented
            
        return -loglike/self.T
    
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
            params = list(self.a)+list(self.b)
            if method == "default":
                llk = self.DCC_loglike(params, complete=complete)
            elif method == "2MSCLE":
                llk = self.DCC_loglike_2MSCLE(params, complete=complete)
            else:
                print("Loglikelihood method unknown")
                
        if self.dist == "t":
            params = list(self.a)+list(self.b)+[self.dof]
            if method == "default":
                llk = self.DCC_logliket(params, complete=complete)
            elif method == "2MSCLE":
                llk = self.DCC_logliket_2MSCLE(params, complete=complete)
            else:
                print("Loglikelihood method unknown")
        
        if self.dist == "skewt":
            params = list(self.a)+list(self.b)+[self.dof] + list(self.c)
            if method == "default":
                llk = self.DCC_loglikeskewt(params, complete=complete)
            elif method == "2MSCLE":
                llk = self.DCC_loglikeskewt_2MSCLE(params, complete=complete)
            else:
                print("Loglikelihood method unknown")
        
        return llk
        
    def predict(self, seq_length, continuity_index = None):
        """
            continuity_index:   if None, intialization with omega and Q_bar
                                if int, will use et[continuity_index], r_t[continuity_index], D_t[continuity_index] and Q_t[continuity_index] to initialize

        """
        gen_D_t = np.zeros((seq_length,self.N))    
        gen_Q_t = np.zeros((seq_length,self.N,self.N))
        gen_R_t = np.zeros((seq_length,self.N,self.N))
        gen_H_t = np.zeros((seq_length,self.N,self.N))
        gen_rt = np.zeros((seq_length, self.N))
        
        omega = self.garch_params[:,0]
        alpha = self.garch_params[:,1:1+self.p1]
        beta = self.garch_params[:,1+self.p1:1+self.p1+self.q1]        
        
        # Initialization
        if continuity_index == None:
            gen_D_t[0,:] = np.sqrt(omega)
            gen_Q_t[0,:] = self.Q_bar
            qts = np.sqrt(np.diag(1/np.diag(gen_Q_t[0])))
            gen_R_t[0,:] = qts @ gen_Q_t[0,:] @ qts
            gen_H_t[0,:] = np.diag(gen_D_t[0,:]) @ gen_R_t[0,:] @ np.diag(gen_D_t[0,:])
        else:
            i = continuity_index
            gen_D_t[0,:] = np.sqrt((1 + alpha[:,:-(i-max(i-self.p1,0))].sum(axis=1) + beta[:,:-(i-max(i-self.q1,0))].sum(axis=1))*omega + (alpha[:,-(i-max(i-self.p1,0)):].T*(self.rt[max(i-self.p1,0):i]**2)).sum(axis=0) + (beta[:,-(i-max(i-self.q1,0)):].T*self.D_t[max(i-self.q1,0):i]**2).sum(axis=0))
            gen_Q_t[0,:] = (1-self.a[:-(i-max(i-self.p2,0))].sum()-self.b[:-(i-max(i-self.q2,0))].sum())*self.Q_bar + (self.a[-(i-max(i-self.p2,0)):, None, None]*(self.et[max(i-self.p2,0):i,:,np.newaxis]*self.et[max(i-self.p2,0):i,np.newaxis,:])).sum(axis=0) + (b[-(i-max(i-self.q2,0)):, None, None]*Q_t[max(i-self.q2,0):i]).sum(axis=0)
            
            qts = np.sqrt(np.diag(1/np.diag(gen_Q_t[0])))
            gen_R_t[0,:] = qts @ gen_Q_t[0,:] @ qts
            gen_H_t[0,:] = np.diag(gen_D_t[0,:]) @ gen_R_t[0,:] @ np.diag(gen_D_t[0,:])
        
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
            gen_Q_t[i,:] = (1-self.a[:-(i-max(i-self.p2,0))].sum()-self.b[:-(i-max(i-self.q2,0))].sum())*self.Q_bar + (self.a[-(i-max(i-self.p2,0)):, None, None]*(gen_et[max(i-self.p2,0):i,:,np.newaxis]*gen_et[max(i-self.p2,0):i,np.newaxis,:])).sum(axis=0) + (self.b[-(i-max(i-self.q2,0)):, None, None]*gen_Q_t[max(i-self.q2,0):i]).sum(axis=0)
            
            
            qts = np.sqrt(np.diag(1/np.diag(gen_Q_t[i])))
            gen_R_t[i,:] = qts @ gen_Q_t[i,:] @ qts
            gen_H_t[i,:] = np.diag(gen_D_t[i,:]) @ gen_R_t[i,:] @ np.diag(gen_D_t[i,:])
            gen_rt[i,:] = sp.linalg.sqrtm(gen_H_t[i,:]) @ gen_et[i,:]
        
        return gen_rt + self.mean[np.newaxis,:], gen_H_t
            
            
    def fit(self, returns, method=None, verbose = True):
        if verbose:
            print("Calibration of DCC("+str(self.p2)+","+str(self.q2)+"), univariate GARCH("+str(self.p1)+","+str(self.q1)+"), distrib "+self.dist+", method "+method)
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
        
        # DCC(1,1) calibration
        if self.dist == 'norm':
            start_params = [0.05/self.p2]*self.p2+[0.7/self.q2]*self.q2
            first_likelihood = self.DCC_loglike(start_params, complete=True)
            
            if verbose:
                print("First neg-llikelihood:\t", first_likelihood)
            
            if method == "2MSCLE":
                res = minimize(self.DCC_loglike_2MSCLE, start_params,
                bounds = [(1e-6, 1-1e-6)]*(self.p2+self.q2), constraints = LinearConstraint(np.ones((1,self.p2+self.q2)), ub=1-1e-2), 
                tol=1e-6, method='SLSQP',
                options = {'maxiter':10000000, 'disp':True},
                )
            else:
                res = minimize(self.DCC_loglike, start_params,
                bounds = [(1e-6, 1-1e-6)]*(self.p2+self.q2), constraints = LinearConstraint(np.ones((1,self.p2+self.q2)), ub=1-1e-2), 
                tol=1e-6, method='SLSQP',
                options = {'maxiter':10000000, 'disp':True},
                )
            self.a = res.x[:self.p2]
            self.b = res.x[self.p2:]
            
            if verbose:
                last_likelihood = self.DCC_loglike(res.x, complete=True)
                print("Last neg-llikelihood:\t", last_likelihood)
                
            return {'mu': self.mean, 'alpha': self.a, 'beta': self.b} 
        elif self.dist == 't':
            start_params = [0.05/self.p2]*self.p2+[0.7/self.q2]*self.q2+[5]
            first_likelihood = self.DCC_logliket(start_params, complete=True)
            const = np.ones((1,self.p2+self.q2+1))
            const[:,-1] = 0
            
            if verbose:
                print("First neg-llikelihood:\t", first_likelihood)
                
            if method == "2MSCLE":
                res = minimize(self.DCC_logliket_2MSCLE, start_params,
                bounds = [(1e-6, 1-1e-6)]*(self.p2+self.q2)+[(3, None)], constraints = LinearConstraint(const, ub=1-1e-2), 
                tol=1e-6, method='SLSQP',
                options = {'maxiter':10000000, 'disp':True},
                )
            else:
                res = minimize(self.DCC_logliket, start_params,
                bounds = [(1e-6, 1-1e-6)]*(self.p2+self.q2)+[(3, None)], constraints = LinearConstraint(const, ub=1-1e-2), 
                tol=1e-6, method='SLSQP',
                options = {'maxiter':10000000, 'disp':True},
                )
            self.a = res.x[:self.p2]
            self.b = res.x[self.p2:-1]
            self.dof = res.x[-1]
            
            if verbose:
                last_likelihood = self.DCC_logliket(res.x, complete=True)
                print("Last neg-llikelihood:\t", last_likelihood)
                
            return {'mu': self.mean, 'alpha': self.a, 'beta': self.b, 'dof': self.dof} 
        
        elif self.dist == 'skewt':
            start_params = [0.05/self.p2]*self.p2+[0.7/self.q2]*self.q2+[10]+ list(np.random.normal(size=self.N)/100)
            first_likelihood = self.DCC_loglikeskewt(start_params, complete=True)
            const = np.ones((1,self.p2+self.q2+1+self.N))
            const[:,-1-self.N] = 0
            
            if verbose:
                print("First neg-llikelihood:\t", first_likelihood)
            
            if method == "2MSCLE":
                res = minimize(self.DCC_loglikeskewt_2MSCLE, start_params,
                bounds = [(1e-6, 1-1e-6)]*(self.p2+self.q2)+[(3, None)] + [(None, None)]*self.N, constraints = LinearConstraint(const, ub=1-1e-2), 
                tol=1e-6, method='SLSQP',
                options = {'maxiter':10000000, 'disp':True},
                )
            else:
                res = minimize(self.DCC_loglikeskewt, start_params,
                bounds = [(1e-6, 1-1e-6)]*(self.p2+self.q2)+[(3, None)] + [(None, None)]*self.N, constraints = LinearConstraint(const, ub=1-1e-2), 
                tol=1e-6, method='SLSQP',
                options = {'maxiter':10000000, 'disp':True},
                )
            
            self.a = res.x[:self.p2]
            self.b = res.x[self.p2:self.p2 + self.q2]
            self.dof = res.x[self.p2 + self.q2]
            self.c = np.array(res.x[self.p2 + self.q2 + 1:])
            
            if verbose:
                last_likelihood = self.DCC_loglikeskewt(res.x, complete=True)
                print("Last neg-llikelihood:\t", last_likelihood)
            return {'mu': self.mean, 'alpha': self.a, 'beta': self.b, 'dof': self.dof, 'c': self.c} 
        else:
            raise NotImplementedError("Model non supported for fitting")