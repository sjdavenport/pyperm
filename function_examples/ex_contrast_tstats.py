import numpy as np
import pyrft as pr

Dim = (10,10); N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
X = pr.group_design(categ); C = np.array([[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(Dim,N)

Dim = (3,3); N = 30; categ = np.zeros(N)
X = pr.group_design(categ); C = np.array(1); lat_data = pr.wfield(Dim,N)

#Need to check that the dimensions of X and C and lat_data match!

# Ensure C is a numpy array
if type(C) != np.ndarray:
    raise Exception("C must be a numpy array")
    
# Ensure that C is a numpy matrix
if len(C.shape) == 0:
    C = np.array([[C]])
elif len(C.shape) == 1:
    C = np.array([C])
elif len(C.shape) > 2:
    raise Exception("C must be a matrix not a larger array")
    
# Calculate the number of contrasts and parameters in C
L = C.shape[0]  # constrasts
C_p = C.shape[1] # parameters

# Calculate the number of parameters p and subjects N
N = X.shape[0] # subjects
p = X.shape[1] # parameters
 
# Ensure that the dimensions of X and C are compatible
if p != C_p:
    raise Exception('The dimensions of X and C do not match')
  
# Ensure that the dimensions of X and lat_data are compatible
if N != lat_data.fibersize:
    raise Exception('The number of subjects in of X and lat_data do not match')

#rfmate = np.identity(p) - np.dot(X, np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)))
# Calculate (X^TX)^(-1)
XTXinv = np.linalg.inv(X.T @ X) 

# Calculate betahat (note leave the extra shaped 1 in as will be remultipling
# with the contrast vectors!)
betahat = XTXinv @ X.T @ lat_data.field.reshape( lat_data.fieldsize + (1,) ) 

# Calculate the residual forming matrix
rfmate = np.identity(N) - X @ XTXinv @ X.T

# Compute the estimate of the variance via the residuals (I-P)Y
# Uses a trick adding (1,) so that multiplication is along the last column!
# Note no need to reshape back not doing so will be useful when dividing by the std
residuals = rfmate @ lat_data.field.reshape( lat_data.fieldsize + (1,) ) 

# Square and sum over subjects to calculate the variance
# This assumes that X has rank p!
std_est = (np.sum(residuals**2,lat_data.D)/(N-p))**(1/2)

# Compute the t-statistics
tstats = (C @ betahat).reshape(lat_data.masksize + (L,))/std_est

# Scale by the scaling constants to ensure var 1
for l in np.arange(L):
    scaling_constant = (C[l,:] @ XTXinv @ C[l,:])**(1/2)
    print(scaling_constant)
    tstats[...,l] = tstats[...,l]/scaling_constant
    
#tstats = ((1/scaling_constants) @ ((C @ betahat))).reshape(lat_data.masksize + (L,))/std_est

# Generate the field of tstats
tstat_field = pr.Field(tstats, lat_data.mask)
print(tstat_field.field)
tstat, xbar, std_dev = pr.mvtstat(lat_data.field)
print(tstat)

# %%
from matplotlib import pyplot as plt 
Dim = (1000,1000); N = 6; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
X = pr.group_design(categ); C = np.array([[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(Dim,N)
p = C.shape[1]

c_tstats = pr.contrast_tstats(lat_data, X, C)
# plt.hist(np.ravel(c_tstats.field), bins = (np.arange(200) -100)/10)

fig, ax = plt.subplots(1, 1); df = 3
x = np.linspace(-5,5, 100)
ax.plot(x, t.pdf(x, df),'r-', lw=5, alpha=0.6, label='t pdf')
ax.hist(np.ravel(c_tstats.field), density=True, histtype='stepfilled', alpha=0.2, bins = x)