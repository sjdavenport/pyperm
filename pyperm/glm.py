import numpy as np
import numpy.matlib
import pyperm as pr
import statsmodels.api as sm

import statsmodels.api as sm


def glm_seq(y, X, distbn, linkfn):
    """ glm_seq runs the generalized linear model sequentially at each voxel in
     the array y.
    Inputs:

        y: a numpy array of shape (n_samples, n_voxels) representing the
            dependent variable
        X: a numpy array of shape (n_samples, n_parameters) representing the
            independent variables
        distbn: a string specifying the distribution of the response variable.
            Must be one of ['Binomial', 'Gamma', 'Gaussian', 'Inverse Gaussian'
                                            , 'Negative Binomial', 'Poisson']
        linkfn: a string specifying the link function. Must be one of ['log',
                'logit', 'probit', 'cauchy', 'cloglog', 'identity', 'inverse']
    Output:
       betahat: a numpy array of shape (n_parameters, n_voxels) containing the
                                        estimated coefficients for each voxel

    Examples:
    -----------------
    # Single voxel example
    nsubj = 50  # number of observations
    nparameters = 5  # dimension of each observation
    # (minus 1 because we have to add the intercept)
    X = np.random.randn(nsubj, nparameters-1)
    intercept = np.ones(nsubj).reshape(nsubj, 1)
    X = np.concatenate((intercept, X), axis=1)
    gamma = np.random.randn(nparameters)
    p = pr.sigmoid(X @ gamma)
    y = np.random.binomial(1, p)
    gammahat, fitted_values = pr.glm_seq(y, X, 'Binomial', 'logit')
    print(gammahat.T)
    print(gamma)

    # Single voxel example
    nsubj = 1000  # number of observations
    nparameters = 5  # dimension of each observation
    # (minus 1 because we have to add the intercept)
    X = np.random.randn(nsubj, nparameters-1)
    intercept = np.ones(nsubj).reshape(nsubj, 1)
    X = np.concatenate((intercept, X), axis=1)
    gamma = np.random.randn(nparameters)
    p = pr.sigmoid(X @ gamma)
    y = np.random.binomial(1, p)
    gammahat, fitted_values = pr.glm_seq(y, X, 'Binomial', 'logit')
    print(gammahat.T)
    print(gamma)

    # Multiple voxel example
    nvoxels = 2
    nsubj = 1000
    nparameters = 5  # dimension of each observation
    # (minus 1 because we have to add the intercept)
    X = np.random.randn(nsubj, nparameters-1)
    intercept = np.ones(nsubj).reshape(nsubj, 1)
    X = np.concatenate((intercept, X), axis=1)
    gamma = np.random.randn(*(nvoxels, nparameters)).T
    p = pr.sigmoid(X @ gamma)
    y = np.random.binomial(1, p)
    gammahat, fitted_values = glm_seq(y, X, 'Binomial', 'logit')

    # Many voxel example
    nvoxels = 10000
    nsubj = 100
    nparameters = 5  # dimension of each observation
    # (minus 1 because we have to add the intercept)
    X = np.random.randn(nsubj, nparameters-1)
    intercept = np.ones(nsubj).reshape(nsubj, 1)
    X = np.concatenate((intercept, X), axis=1)
    gamma = np.random.randn(*(nvoxels, nparameters)).T
    p = pr.sigmoid(X @ gamma)
    y = np.random.binomial(1, p)
    gammahat, fitted_values = glm_seq(y, X, 'Binomial', 'logit')
"""
    # Obtain the sm link function
    link_fn = getattr(sm.families.links, linkfn)
    link = link_fn()

    family_fn = getattr(sm.families, distbn)
    family = family_fn(link=link)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Initialize an empty array to store the estimated coefficients
    betahat = np.empty((X.shape[1], y.shape[1]))

    # Initialize an empty array to store the fitted values
    fitted_values = np.empty((y.shape[0], y.shape[1]))

    # Initialize an empty array to store the confidence intervals
    conf_ints = np.empty((X.shape[1], y.shape[1], 2))

    # Loop over the columns of y
    for i in range(y.shape[1]):
        # Fit a GLM for the i-th column of y
        model = sm.GLM(y[:, i], X, family=family)
        result = model.fit()

        # Extract the coefficients and store them in betahat
        betahat[:, i] = result.params

        # Extract the fitted values and store them in fitted_values
        fitted_values[:, i] = result.fittedvalues

        # Store the confidence intervals
        conf_ints[:, i, :] = result.conf_int()

    return betahat, fitted_values


def run_glm(y, X, nruns=100, learning_rate=0.01):
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
    nparameters = 5 # dimension of each observation
    X = np.random.randn(nsubj, nparameters-1) # (minus 1 because we have to add the intercept)
    intercept = np.ones(nsubj).reshape(nsubj, 1) 
    X = np.concatenate((intercept, X), axis = 1)
    beta = np.random.randn(nparameters)
    p = pr.sigmoid(X @ beta)
    y = np.random.binomial(1, p)
    pr.run_glm(y, X)

    # Multiple voxel example
    nvoxels = 2
    nsubj = 1000
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
    # initialize betahat estimates
    betahat = np.random.randn(*(nparameters, nvoxels))
    fitted_values = sigmoid(X @ betahat)  # initialize p_hats

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
        def var_fn(x): return 0*x + 1
    elif family == 'binomial':
        def var_fn(x): return x*(1-x)
    elif family == 'poisson':
        def var_fn(x): return x
    elif family == 'gamma':
        def var_fn(x): return x*x
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
        def link_fn(x): return x
        def link_fn_deriv(x): return 1
        def link_inv(x): return x
    elif link == 'log':
        def link_fn(x): return np.exp(x)
        def link_fn_deriv(x): return np.exp(x)
        def link_inv(x): return np.log(x)
    elif link == 'logit':
        def link_fn(x): return np.exp(x)/(1 + np.exp(x))
        def link_fn_deriv(x): return np.exp(x)/(1 + np.exp(x)) - \
            np.exp(x)*np.exp(x)/(1 + np.exp(x))**2

        def link_inv(x): return 1 / (1 + np.exp(-x))
    elif link == 'cloglog':
        def link_fn(x): return 1 - np.exp(-np.exp(x))
        def link_fn_deriv(x): return np.exp(-np.exp(x))*np.exp(x)
        # link_inv =
    elif link == 'probit':
        def link_fn(x): return np.exp(-x ^ 2/2)/np.sqrt(2*np.pi)
        # link_fn_deriv =
        # link_inv =
    elif link == 'cauchit':
        def link_fn(x): return 1/(np.pi*(1+x**2))
        # link_fn_deriv =
        # link_inv =
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
    # chatgpt
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
