import statsmodels.api as sm
import numpy as np

# Load the data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]) # feature matrix
y = np.array([0, 1, 0, 1]) # response vector

# Add a column of ones to the feature matrix (for the intercept term)
X = sm.add_constant(X)

# Specify the GLM model
model = sm.GLM(y, X, family=sm.families.Binomial())

# Fit the model to the data
results = model.fit()

# Print the coefficients and intercept
print(results.params)


def run_glm_newton(y, X, nruns=100):
    """ run_glm_newton runs the generalized linear model with data y using Newton-Raphson

    Parameters
    -----------------
    y:   an nsubjects by nvoxels numpy array
    X:   a design matrix of size nsubjects by nparameters
    nruns: int,
        giving the number of runs, default is 100

    Returns
    -----------------
    betahat: np.array,
        of size nparameters by nvoxels giving the estimates of the model parameters
    fitted_values:  np.array,
        of size nsubjects by nvoxels giving the fitted values of the model, 
        i.e. g^(-1)(Xbetahat) where g is the link function

    Examples
    -----------------
    # Single Voxel example
    nsubj = 1000  # number of observations
    nparameters = 10 # dimension of each observation
    X = np.random.randn(nsubj, nparameters-1) # (minus 1 because we have to add the intercept)
    intercept = np.ones(nsubj).reshape(nsubj, 1) 
    X = np.concatenate((intercept, X), axis = 1)
    beta = np.random.randn(nparameters)
    p = pr.sigmoid(X @ beta)
    y = np.random.binomial(1, p)
    run_glm_newton(y, X)
    
    # Multiple voxel example
    nvoxels = 10
    nsubj = 100
    nparameters = 10 # dimension of each observation
    X = np.random.randn(nsubj, nparameters-1) # (minus 1 because we have to add the intercept)
    intercept = np.ones(nsubj).reshape(nsubj, 1) 
    X = np.concatenate((intercept, X), axis = 1)
    beta = np.random.randn(*(nvoxels, nparameters))
    p = pr.sigmoid(X @ beta.T)
    y = np.random.binomial(1, p)
    betahat, fitted_values = run_glm_newton(y, X)
    """
    nparameters = X.shape[1]
    betahat = np.random.randn(nparameters)
    nvoxels = y.shape[1]
    betahat = np.random.randn(*(nparameters, nvoxels)) # initialize betahat estimates
    fitted_values = sigmoid(X @ betahat) # initialize p_hats

    for run in range(nruns):
        # get Hessian
        a = fitted_values*(1-fitted_values)
        W = np.diag(a.flatten())
        Hessian = X.T @ W @ X

        # get gradient 
        a = y*(1-fitted_values)
        b = (1-y)*fitted_values 
        grad = X.T @ (b-a)
        
        # adjust beta hats
        betahat -= np.linalg.inv(Hessian) @ grad
        
        # adjust p hats
        fitted_values = sigmoid(X @ betahat)
    
    betahat = betahat.T
        
    return betahat, fitted_values
