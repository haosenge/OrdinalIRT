import pyximport
pyximport.install(reload_support=True)

import truncnormal
import cProfile
from timeit import default_timer as timer
from scipy.stats import truncnorm
import numpy as np
import scipy.special as cs
import matplotlib.pyplot as plt
import math

from importlib import reload
reload(truncnormal)


# =============================================================================
# import importlib
# importlib.reload('truncnormal')        
# =============================================================================

def tuncnormal(loc, a,  b, scale, size):
    
    z = np.zeros(size)
    
    
    for idx in range(size):
        
        a_norm = (a[idx] - loc[idx]) / scale
        b_norm = (b[idx] - loc[idx]) / scale
        
        # Inverse Transformation Sampling
        if (b_norm - a_norm >= 1) and (not (a_norm < -4 and b_norm < -4)) and (not (a_norm > 4 and b_norm > 4)):
            u = np.random.uniform(size = 1, low = cs.ndtr(a_norm), high = cs.ndtr(b_norm))[0]
            z_norm = cs.ndtri(u)
            z[idx] = z_norm * scale + loc[idx]
            
        # Uniform Proposal Rejection Sampling
        elif b_norm - a_norm < 1: 
            if b_norm * a_norm < 0:
                m = 1 / np.sqrt(2 * math.pi) 
            else:
                m = np.maximum(1/np.sqrt(2 * math.pi) * np.exp(-0.5 * (a_norm) ** 2), 1/np.sqrt(2 * math.pi) * np.exp(-0.5 * (b_norm) ** 2))
            
            p = 0
            while p < 1:
                
                u = np.random.uniform(size = 1, low = a_norm, high = b_norm)[0]
                v = np.random.uniform(size = 1, low = 0, high = 1)[0]
                
                if v < (1/np.sqrt(2 * math.pi) * np.exp(-0.5 * (u) ** 2)) / m:
                    z[idx] =  u * scale + loc[idx]
                    p = 1
                    
        # Rayleigh Proposal Rejection Sampling
        else:
            flag = 0
            
            if b_norm < 0:
                # Note the flip of relationship when times -1
                a_norm = (b[idx] - loc[idx]) / scale * (-1)
                b_norm = (a[idx] - loc[idx]) / scale * (-1)
                flag = 1
            
            c = a_norm ** 2 / 2
            q = 1 - np.exp(c - (b_norm ** 2 / 2))
            
            p = 0
            while p < 1:
                
                u = np.random.uniform(size = 1, low = 0, high = 1)[0]
                v = np.random.uniform(size = 1, low = 0, high = 1)[0]
                z_norm = c - np.log(1 - q * u)
                
                
                if z_norm * v ** 2 <= a_norm:
                    if flag == 1:
                        z[idx] = (scale * (np.sqrt(2 * z_norm)) - loc[idx]) * (-1)
                    else:
                        z[idx] = scale * (np.sqrt(2 * z_norm)) + loc[idx]
                    p = 1
            
    return z


mu = np.full(100000,0)
a = np.full(100000, -np.inf)
b = np.full(100000, 3)


start = timer()
sample = ct.tuncnormal(mu, a, b, 1, len(mu))
end = timer()
print(end - start)

_ = np.linspace(-10, b[0]+0.5, 100000)
t = truncnorm.pdf(_, a = a[0], b = b[0])
plt.plot(_, t)
plt.hist(sample, density = True, bins = 20)
plt.show()


tuncnormal(theta_single[one_mask], 
            np.full(sum(one_mask.flatten()), -np.inf),
            alpha_R[one_mask],
            1,
            sum(one_mask.flatten()))