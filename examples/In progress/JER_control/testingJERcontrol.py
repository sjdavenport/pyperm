"""
Testing the JER control over contrasts on some examples
"""
import numpy as np
import pyrft as pr
import sanssouci as ss

Dim = (10,10); N = 30; categ = np.random.binomial(1, 0.4, size = N)
X = pr.groupX(categ); C = np.array([[1,-1]]); lat_data = pr.wfield(Dim,N)
signal = 0.5

w0=np.where(categ==0)
# Add  to the data when the category is 1! :)
lat_data.field[:,:,w0] = lat_data.field[:, :, w0] + signal

B = 1000

m = np.prod(Dim)

minPperm, orig_pvalues, pivotal_stats = pr.boot_contrasts(lat_data, X, C, B)

# Choose the confidence level
alpha = 0.1

# Obtain the lambda calibration
lambda_quant = np.quantile(pivotal_stats, alpha)

# Gives t_k^L(lambda) = lambda*k/m for k = 1, ..., m
thr = ss.t_linear(lambda_quant, np.arange(1,m+1), m)

# Get the first 10 pvalues
pvals = np.sort(np.ravel(orig_pvalues.field))[:10]

# Compute an upper bound on this
bound = ss.max_fp(pvals, thr)
print(bound)

# %% Testing the JER control (ensuring that the JER is corrrectly controlled)
niters = 1000
alpha = 0.1
B = 100

Dim = (10,10); N = 30; C = np.array([[1,-1]]); 
m = np.prod(Dim)
signal = 1

nbelow = 0 # Initialize the FPR counter

for I in np.arange(niters):
    print(I)
    
    categ = np.random.binomial(1, 0.4, size = N)
    X = pr.groupX(categ); lat_data = pr.wfield(Dim,N)
    # If you want to test it when you add signal!
    # w0=np.where(categ==0)
    # lat_data.field[:,:,w0] = lat_data.field[:, :, w0] + signal
    
    minPperm, orig_pvalues, pivotal_stats = pr.boot_contrasts(lat_data, X, C, B)
    lambda_quant = np.quantile(pivotal_stats, alpha)

    if pivotal_stats[0] < lambda_quant:
        nbelow = nbelow + 1

FPR = nbelow/niters

# %% Testing the JER control (global null)
niters = 1000
alpha = 0.1
B = 100

Dim = (20,20); N = 30; 
from sklearn.utils import check_random_state
rng = check_random_state(101)
C = np.array([[1,-1,0],[0,1,-1]]); 

m = np.prod(Dim)

nbelow = 0 # Initialize the FPR counter

for I in np.arange(niters):
    print(I)
    
    categ = rng.choice(3, N, replace = True)
    X = pr.groupX(categ); lat_data = pr.wfield(Dim,N)
    # If you want to test it when you add signal!
    # w0=np.where(categ==0)
    # lat_data.field[:,:,w0] = lat_data.field[:, :, w0] + signal
    
    minPperm, orig_pvalues, pivotal_stats = pr.boot_contrasts(lat_data, X, C, B)
    lambda_quant = np.quantile(pivotal_stats, alpha)

    if pivotal_stats[0] < lambda_quant:
        nbelow = nbelow + 1

FPR = nbelow/niters

# %% Testing the JER control (strong control)
niters = 1000
alpha = 0.1
B = 100

Dim = (10,10); N = 30; 
from sklearn.utils import check_random_state
rng = check_random_state(101)
C = np.array([1,-1,0]);  lat_data = pr.wfield(Dim,N)
w2 = np.where(categ == 2)
minP, orig_pvalues, pivotal_stats = pr.boot_contrasts(lat_data, X, C)

N = 30; C = np.array([[1,-1]]); 
m = np.prod(Dim)
signal = 1

nbelow = 0 # Initialize the FPR counter

for I in np.arange(niters):
    print(I)
    
    categ = rng.choice(3, N, replace = True)
    X = pr.groupX(categ); lat_data = pr.wfield(Dim,N)
    lat_data.field[:,:,w2]=  lat_data.field[:,:,w2] + signal
    
    minPperm, orig_pvalues, pivotal_stats = pr.boot_contrasts(lat_data, X, C, B)
    lambda_quant = np.quantile(pivotal_stats, alpha)

    if pivotal_stats[0] < lambda_quant:
        nbelow = nbelow + 1

FPR = nbelow/niters