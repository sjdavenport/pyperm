import numpy as np 
import numpy.matlib
import pyperm as pr
import statsmodels.api as sm

#def compute_scores(y, X, Z, family, link, score_type = 'effective', nruns = 100, learning_rate = 0.01):
def compute_scores(model0, model1, score_type = 'effective'):
    nsubj = y.shape[0]
    
    # Initialize the weights matrix to be ones for now - this may change!
    W = np.matlib.repmat(1,1,nsubj)
    W = W[0]
    W = W[:,None]
    sqrtW = np.sqrt(W)
    
    # Obtain the derivative and variance evaluated at the fitted values
    D, V = get_par_expo_fam(fitted_values_0, family, link)
    
    # Compute the inverse square root of V
    sqrtinvV = V**(-0.5)
    
    # Multiple this by the residuals
    sqrtinvV_times_residuals=sqrtinvV_vect*residuals
    
    #if score_type == 'effective'
    #    A = Z*sqrtW
      #  B = X*sqrtW
       # H = A @ np.linalg.inv(A.T @ A) @ A
        #scores = B*sqrtinvV_vect_times_residuals/(length(model0$y)**0.5)


#B = np.transpose(np.transpose(X * D_vect) @ (np.diag(np.sqrt(1 / V_vect**2), nrow=len(sqrtW))))
#scores = B * (sqrtinvV_vect_times_residuals) * (1/len(model0['y'])**0.5)
#scale_objects = {'nrm': np.sqrt(np.sum(B**2) * np.sum((sqrtinvV_vect_times_residuals)**2) / len(model0['y']))}
#result = {'scores': scores, 'scale_objects': scale_objects}


def run_glm(y, X, nruns = 100, learning_rate = 0.01):
    """ run_glm runs the generalized linear model with data y

    Parameters
    -----------------
    y:   an nsubjects by nvoxels numpy array
    X:   a design matrix of size nsubjects by nparameters
    nruns: int,
        giving the number of runs, default is 100
    learning_rate: float,
        giving the rate at which gradient descent is applied. Default is 0.01.
        Could investigate varying this as the procedure continues for speed.

    Returns
    -----------------
    betahat:
    fitted_values:  np.array,
        of size nsubjects by nvoxels giving the fitted values of the model, 
        i.e. g^(-1)(Xbetahat) where g is the link function
    grad:
        

    Examples
    -----------------
    # Single Voxel example
    nsubj = 1000  # number of observations
    nparameters = 10 # dimension of each observation
    X = np.random.randn(nsubj, nparameters-1) # (minus 1 because we have to add the intercept)
    intercept = np.ones(nsubj).reshape(nsubj, 1) 
    X = np.concatenate((intercept, X), axis = 1)
    beta = np.random.randn(nparameters)
    p = sigmoid(X @ beta)
    y = np.random.binomial(1, p)
    run_glm(y, X)
    
    # Multiple voxel example
    nvoxels = 10000
    nsubj = 250
    nparameters = 10 # dimension of each observation
    X = np.random.randn(nsubj, nparameters-1) # (minus 1 because we have to add the intercept)
    intercept = np.ones(nsubj).reshape(nsubj, 1) 
    X = np.concatenate((intercept, X), axis = 1)
    beta = np.random.randn(*(nvoxels, nparameters))
    p = pr.sigmoid(X @ beta.T)
    y = np.random.binomial(1, p)
    betahat, fitted_values, grad = pr.run_glm(y, X)
    
    More subjects doesn't necessarily require more time as the convergence is 
    quicker so the number of runs needed is in fact less!
    
    """
    nparameters = X.shape[1]
    betahat = np.random.randn(nparameters)
    nvoxels = y.shape[1]
    betahat = np.random.randn(*(nparameters, nvoxels)) # initialize betahat estimates
    fitted_values = sigmoid(X @ betahat) # initialize p_hats
    
    for run in range(nruns):
        # get gradient 
        a = y*(1-fitted_values)
        b = (1-y)*fitted_values 
        grad = X.T @ (b-a)
        
        # adjust beta hats
        betahat -= learning_rate*grad
        
        # adjust p hats
        fitted_values = sigmoid(X @ betahat)
        
    betahat = betahat.T
        
    return betahat, fitted_values, grad
    
def get_par_expo_fam(fitted_values, family, link):
    """
    Parameters:
        fitted_values (array-like): the fitted values for which to compute the 
                                    derivative and variance
        family (str): the exponential family to use, can be 'gaussian', 
                    'binomial', 'poisson', or 'gamma'
        link (str): the link function to use, can be 'identity', 'log', 'logit',
                    'cloglog', 'probit', or 'cauchit'

    Returns:
        tuple: a tuple containing the derivative of the link function and the 
        variance function at the fitted values.
    """
    if family == 'gaussian':
        var_fn = lambda x: 0*x + 1
    elif family == 'binomial':
        var_fn = lambda x: x*(1-x)
    elif family == 'poisson':
        var_fn = lambda x: x
    elif family == 'gamma':
        var_fn = lambda x: x*x
    else:
        raise Exception('The family must be one of the specified ones')
        
    # Compute the variance funciton at the fitted values
    Vhat = var_fn(fitted_values)
    
    # Get the derivative of the link function
    _, link_fn_deriv, _ = get_linkfn(link)
    
    # Compute the derivative of the link at the fitted values
    Dhat = link_fn_deriv(fitted_values)
    
    return Dhat, Vhat

def get_linkfn(link):
    if link == 'identity':
        link_fn = lambda x: x
        link_fn_deriv = lambda x: 1
        link_inv = lambda x: x
    elif link == 'log':
        link_fn = lambda x: np.exp(x)
        link_fn_deriv = lambda x: np.exp(x)
        link_inv = lambda x: np.log(x)
    elif link == 'logit':
        link_fn = lambda x: np.exp(x)/(1 + exp(x))
        link_fn_deriv = lambda x: np.exp(x)/(1 + np.exp(x)) - np.exp(x)*np.exp(x)/(1 + np.exp(x))**2
        link_inv = lambda x: 1 / (1 + np.exp(-x))
    elif link == 'cloglog':
        link_fn = lambda x: 1 - np.exp(-np.exp(x))
        link_fn_deriv = lambda x: np.exp(-np.exp(x))*exp(x)
        #link_inv = 
    elif link == 'probit':
        link_fn = lambda x: np.exp(-x^2/2)/sqrt(2*pi)
        #link_fn_deriv =
        #link_inv = 
    elif link == 'cauchit':
        link_fn = lambda x: 1/(np.pi*(1+x**2))
        #link_fn_deriv =
        #link_inv = 
    else:
        raise Exception('The link function must be one of the specified ones')
    
    return link_fn, link_fn_deriv, link_inv

def sigmoid(x): return 1 / (1 + np.exp(-x))


def fit_generalized_linear_model(formula, data, link, family):
    """
    Fits a generalized linear model with the given formula and data.
    
    Parameters:
    formula (str): a string specifying the model in the form of 
                  "response ~ predictor1 + predictor2 + ... + predictorN"
    data (pandas DataFrame): a DataFrame containing the predictor and response variables
    link (statsmodels link): the link function to use in the model
    family (statsmodels family): the distribution family to use in the model
    
    Returns:
    A fitted statsmodels generalized linear model object
    """
    model = sm.GLM.from_formula(formula, data=data, family=family, link=link)
    result = model.fit()
    return result

def logistic_regression_newton_raphson(y, X, nruns=100, epsilon=1e-6):
    #chatgpt
    nparameters = X.shape[1]
    nvoxels = y.shape[1]
    betahat = np.random.randn(*(nparameters, nvoxels))
    fitted_values = sigmoid(X @ betahat)
    niter = 0
    
    for run in range(nruns):
        W = np.diag(fitted_values * (1 - fitted_values))
        grad = X.T @ (y - fitted_values)
        Hessian = X.T @ W @ X
        betahat_new = betahat + np.linalg.inv(Hessian) @ grad
        
        # check for convergence
        if np.abs(betahat_new - betahat).max() < epsilon:
            break
            
        betahat = betahat_new
        fitted_values = sigmoid(X @ betahat)
        niter += 1
    
    return betahat, fitted_values, niter
