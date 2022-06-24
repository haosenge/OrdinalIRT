import numpy as np
import numba
from numpy.linalg import inv
import cython_func.truncnormal as ct
import cython_func.alpha_sampler as alpha

print('model loaded', flush=True)


# Give Observed Data y_observe, estimate (theta, alpha_E, alpha_R)
class DynamicOrdinalIRT:

    def __init__(self, y_observe):
        self.y_observe = y_observe  # I * J * T ndarray
        self.I = y_observe.shape[0]
        self.J = y_observe.shape[1]
        self.T = y_observe.shape[2]
    
    def GibbsSampler(self, init, burnin, iterations):
        
        y_observe = self.y_observe

        # initialization
        theta = init['theta']    # J * T matrix
        alpha_R = init['alpha_R']   # I * J matrix
        alpha_E = init['alpha_E']   # I * J matrix
        
        # Constraint: alpha_E > alpha_R
        if (alpha_E < alpha_R).any():
            raise ValueError('Some alpha_E_ij is less than alpha_R_ij')
        
        result = {'theta': [], 'alpha_R': [], 'alpha_E': []}
        for i in range(iterations):
            
            # Reshape theta to I * J * T for easy manipulation
            theta_ijt = np.zeros((self.I, self.J, self.T))
            for t in range(self.T):
                theta_ijt[:,:,t] = np.tile(theta[:,t], (self.I, 1))
                
            # Step 1: Simulate Y Latent
            y_latent = self.simulate_y_latent(theta_ijt = theta_ijt, alpha_R = alpha_R, alpha_E = alpha_E)
            
            # Step 2: Sample alpha_R
            R_cutoff = self.obtain_r0_r1(y_observe, y_latent, alpha_R, alpha_E, True)
            alpha_R = alpha.sample_alpha(alpha_R.shape[0], alpha_R.shape[1], R_cutoff)
            result['alpha_R'].append(alpha_R.copy())
            
            # Step 3: sample alpha_E
            E_cutoff = self.obtain_r0_r1(y_observe, y_latent, alpha_R, alpha_E, False)
            alpha_E = alpha.sample_alpha(alpha_E.shape[0], alpha_E.shape[1], E_cutoff)
            result['alpha_E'].append(alpha_E.copy())
            
            # Step 4 Sample theta
            theta = self.forward_backward(self.T, self.J, y_latent, theta, self.kalman_filter, self.backward_sampler)
                
            result['theta'].append(theta.copy())
            
            print(i, flush=True)
                
        # output
        return result
    
    def simulate_y_latent(self, theta_ijt, alpha_R, alpha_E):

        y_latent = np.zeros(self.y_observe.shape)

        for time in range(self.T):
            
            y_obs_single = self.y_observe[:, :, time]
            y_latent_single = y_latent[:, :, time]
            theta_single = theta_ijt[:,:,time]
            
            one_mask = y_obs_single == 1
            y_latent_single[one_mask] = ct.tuncnormal(theta_single[one_mask], 
                                                      np.full(sum(one_mask.flatten()), -np.inf),
                                                      alpha_R[one_mask],
                                                      1,
                                                      sum(one_mask.flatten()))
            

            if (y_latent_single[one_mask] > alpha_R[one_mask]).any():
                 raise ValueError('Some latent y is wrong when Y = 1')

            two_mask = y_obs_single == 2
            y_latent_single[two_mask] = ct.tuncnormal(theta_single[two_mask], # mean
                                                      alpha_R[two_mask], # a
                                                      alpha_E[two_mask], # b
                                                      1,    # scale
                                                      sum(two_mask.flatten()))  # size
            

            if ((y_latent_single[two_mask] < alpha_R[two_mask]) | (y_latent_single[two_mask] > alpha_E[two_mask])).any():
                raise ValueError('Some latent y is wrong when Y = 2')

            three_mask = y_obs_single == 3
            y_latent_single[three_mask] = ct.tuncnormal(theta_single[three_mask], # mean
                                                      alpha_E[three_mask], # a
                                                      np.full(sum(three_mask.flatten()), np.inf), # b
                                                      1,    # scale
                                                      sum(three_mask.flatten()))  # size

            if (y_latent_single[three_mask] < alpha_E[three_mask]).any():
                 raise ValueError('Some latent y is wrong when Y = 3')

        return y_latent
      
    # Sample R0 and R1
    @staticmethod
    @numba.njit
    def obtain_r0_r1(y_observe, y_latent, alpha_R, alpha_E, sample_alpha_R):
        
        cutoff_table = np.zeros((alpha_R.shape[0], alpha_R.shape[1], 2)) # d3 = 1 ==> r0, d3 = 2 ==> r1
        # Sample alpha_R
        if sample_alpha_R == True:
            for row_id in range(y_latent.shape[0]):
                for col_id in range(y_latent.shape[1]):
                    
                    latent_time_series = y_latent[row_id, col_id, :]
                    obs_time_series = y_observe[row_id, col_id, :]
                    
                    if latent_time_series[obs_time_series == 1].size > 0:
                        r0 = max(latent_time_series[obs_time_series == 1])
                    else:
                        r0 = -np.inf
                        
                    if latent_time_series[obs_time_series == 2].size > 0:
                        r1 = min(latent_time_series[obs_time_series == 2])
                    else:
                        r1 = alpha_E[row_id, col_id]
                    
                    # Store input
                    cutoff_table[row_id, col_id, 0] = r0
                    cutoff_table[row_id, col_id, 1] = r1
        
        # Sample alpha_E
        else:
            
            for row_id in range(y_latent.shape[0]):
                for col_id in range(y_latent.shape[1]):
                    
                    latent_time_series = y_latent[row_id, col_id, :]
                    obs_time_series = y_observe[row_id, col_id, :]
                    
                    if latent_time_series[obs_time_series == 2].size > 0:
                        r0 = max(latent_time_series[obs_time_series == 2])
                    else:
                        r0 = alpha_R[row_id, col_id]
                        
                    if latent_time_series[obs_time_series == 3].size > 0:
                        r1 = min(latent_time_series[obs_time_series == 3])
                    else:
                        r1 = np.inf
                    
                    # Store input
                    cutoff_table[row_id, col_id, 0] = r0
                    cutoff_table[row_id, col_id, 1] = r1
        
        return cutoff_table
    
    @staticmethod
    @numba.njit
    def kalman_filter(m1, var1, y2):
        
        dim = len(y2)
        
        pred_mu2_mean = m1   # scalar
        pred_mu2_var = var1 + 1   # scalar
        
        pred_y2_mean = np.array([m1] * dim)  # I * 1 vector
        pred_y2_var = np.identity(dim) + np.full((dim, dim), pred_mu2_var) # I * I matrix
        
        mu2_mean = pred_mu2_mean + np.dot(
                                    np.dot(np.full(dim, pred_mu2_var, dtype = 'float64'), inv(pred_y2_var)), 
                                            (y2 - pred_y2_mean))
        mu2_var = pred_mu2_var - np.dot(
                                    np.dot(np.full(dim, pred_mu2_var, dtype = 'float64'), inv(pred_y2_var)), 
                                    np.full(dim, pred_mu2_var, dtype = 'float64'))
        
        return np.array([mu2_mean, mu2_var])
    
    @staticmethod
    @numba.njit
    def backward_sampler(m1, var1, mu2):
        
        pred_mu2_var = var1 + 1
        pred_mu2_mean = m1
        
        mu_post = m1 + var1 * (pred_mu2_var) ** (-1) * (mu2 - pred_mu2_mean)
        var_post = var1 - var1 * (pred_mu2_var) ** (-1) * var1
        
        sample = np.random.normal(loc = mu_post, scale = np.sqrt(var_post))
        
        return sample
    
    @staticmethod
    @numba.njit
    def forward_backward(T, J, y_latent, theta, kalman_filter, backward_sampler):
        
        for country_id in range(J):
            
            if country_id == 0:
                # Prior Information
                m0 = -5
                var0 = 1
            elif country_id == J - 1:
                m0 = 5
                var0 = 1
            else:
                m0 = 0
                var0 = 25
            
            # Define y
            y = y_latent[:,country_id,:]
            
            para_forward = np.zeros((T,2))
            
            for t in range(T):
                if t== 0:
                    para_temp = kalman_filter(m1 = m0, var1 = var0, y2 = y[:,t])
                else:
                    para_temp = kalman_filter(m1 = para_forward[t-1][0], var1 = para_forward[t-1][1], y2 = y[:,t])
                    
                para_forward[t,:] = para_temp
                
                
            # Backward Sampler Pr(mu_t | mu_(t+1), Dt)
            sample_backward = np.zeros(T)
            
            for t in range(T - 1,-1,-1):
                
                if t == T - 1:
                    para_b_temp = np.random.normal(loc = para_forward[t][0], scale = np.sqrt(para_forward[t][1]))
                else:
                    para_b_temp = backward_sampler(m1 = para_forward[t][0], var1 = para_forward[t][1], mu2 = sample_backward[t+1])
    
                sample_backward[t] = para_b_temp
            
            if (np.isnan(sample_backward).any()):
                raise ValueError('Backward sample has NaN')
                
            # Update the theta matrix
            theta[country_id,:] =  sample_backward
            
            # Rescale the first period
            p1_center = np.mean(theta[:,0])
            p1_var = np.var(theta[:,0])
            theta[:,0] = (theta[:,0] - p1_center) / np.sqrt(p1_var)
            
        return theta