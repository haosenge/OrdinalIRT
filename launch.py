import os
import pickle
import numpy as np
from model import DynamicOrdinalIRT

with open('data/estimation_data.pickle', 'rb') as handle:
    irt_data = pickle.load(handle)

data = irt_data['data']

print('data has dimension {}'.format(data.shape), flush=True)

obj = DynamicOrdinalIRT(data)

# Initialization
I = data.shape[0]
J = data.shape[1]
T = data.shape[2]

# Chain 1
alpha_R_p = np.random.normal(size = I * J, loc = 0, scale = 2).reshape(I,J)
alpha_E_p = alpha_R_p + np.random.uniform(size = I * J, low = 0, high = 5).reshape(I,J)
theta_p = np.random.normal(size = J * T, loc = 0, scale = 2).reshape(J,T)
init = {'alpha_R': alpha_R_p, 'alpha_E':alpha_E_p, 'theta': theta_p}

result = obj.GibbsSampler(init = init, burnin = 1000, iterations = 5000)

# Save
print("file save path", os.getcwd(), flush = True)

# Save All data
with open('data/result_data.pickle', 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save Theta
theta_data = {'theta': result['theta']}
with open('data/result_data_theta.pickle', 'wb') as handle:
    pickle.dump(theta_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    