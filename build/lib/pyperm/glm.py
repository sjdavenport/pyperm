import numpy as np
import numpy.matlib
import pyperm as pr
import statsmodels.api as sm
import warnings


def glm_perm(y, Z, X, distbn, link, scores=None, nperm=1000, alpha=0.05):
    """ glm_perm runs sign-flipping pemrutations for generalized linear model
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

    Examples:
    nvoxels = 1000
    nsubj = 30
    nparameters = 2
    X = np.random.randn(nsubj, 1)
    Z = np.ones(nsubj).reshape(nsubj, 1)
    ZX_design = np.concatenate((Z, X), axis=1)
    Z_coeffs = np.random.randn(*(nvoxels, 1)).T
    X_coeffs = np.concatenate(
        (np.tile(0,(1,nvoxels//2)), np.tile(1,(1,nvoxels//2))), axis = 1)
    beta = np.concatenate((Z_coeffs, X_coeffs), axis=0)
    p = pr.sigmoid(ZX_design @ beta)
    y = np.matrix(np.random.binomial(1, p))

    max_vec, significant_locations = pr.glm_perm(y, Z, X, 'binomial', 'logit')

    # Larger realistic example
    nsubj = 238
    nvoxels = 1000
    intercept = np.ones(nsubj)
    age = np.random.poisson(50, nsubj)/50
    sex = np.random.binomial(1, 0.5, nsubj)
    disease_duration = np.random.poisson(15, nsubj)/15
    EDSS = np.random.poisson(100, nsubj)/100
    PASAT = np.random.poisson(100, nsubj)/100
    clinical_subtype = np.random.binomial(1, 0.5, nsubj)

    Z = np.matrix(
        np.stack([intercept, age, sex, disease_duration, EDSS, PASAT], axis=0).T)
    X = np.matrix(clinical_subtype).T
    ZX_design = np.concatenate((Z, X), axis=1)
    Z_coeffs = np.random.randn(*(nvoxels, Z.shape[1])).T
    X_coeffs = np.concatenate(
        (np.tile(0, (1, nvoxels//2)), np.tile(1, (1, nvoxels//2))), axis=1)
    beta = np.concatenate((Z_coeffs, X_coeffs), axis=0)

    p = pr.sigmoid(ZX_design @ beta)
    y = np.matrix(np.random.binomial(1, p))

    max_vec, significant_locations = pr.glm_perm(y, Z, X, 'binomial', 'logit')
    np.sum(significant_locations)
    -----------------
    """
    nsubj = X.shape[0]
    # nvox = y.shape[-1]
    # test_presence = np.sum(y, 0)
    # allzero_indices = np.where(test_presence == 0)[0]
    # allone_indices = np.where(test_presence == nsubj)[0]
    # allvals_indices = np.concatenate((allzero_indices, allone_indices))
    # indices2use = np.setdiff1d(np.arange(nvox), allvals_indices)

    scores, pslocs, glm0, glm1 = pr.compute_scores_mv(y, Z, X, distbn, link)
    ndim = len(scores.shape)
    orig_stat = np.mean(scores, ndim-1)

    max_vec = np.zeros((1, nperm))[0]
    max_vec[0] = np.max(orig_stat)
    print('')
    for i in range(nperm-1):
        pr.progress_bar(i, nperm-1, 'Running permutation, progress:')
        flips = (np.random.binomial(1, 0.5, nsubj) - 0.5)*2
        sign_flipped_scores = np.multiply(scores.copy(), flips)
        fn2evals = np.mean(sign_flipped_scores, ndim-1)
        max_vec[i+1] = np.max(fn2evals)

    max_alpha_quantile = np.quantile(max_vec, 1-alpha)
    significant_locations = orig_stat > max_alpha_quantile
    significant_locations[pslocs] = 0

    return max_vec, significant_locations, scores, glm0, glm1


def compute_scores_mv(y, Z, X, distbn, link):
    """
    Inputs:
        y: a numpy matrix of shape(n_samples, n_voxels) representing the
            dependent variable
        X: a numpy array of shape(n_samples, n_parameters) representing the
            independent variables
        distbn: a string specifying the distribution of the response variable.
            Must be one of['Binomial', 'Gamma', 'Gaussian', 'Inverse Gaussian',
                                               'Negative Binomial', 'Poisson']
        link: a string specifying the link function. Must be one of['log',
                'logit', 'probit', 'cauchy', 'cloglog', 'identity', 'inverse']
    Output:

    Examples:
    -----------------
    nvoxels = 2
    nsubj = 1000
    nparameters = 5  # dimension of each observation
    # (minus 1 because we have to add the intercept)
    X = np.random.randn(nsubj, nparameters-1)
    Z = np.ones(nsubj).reshape(nsubj, 1)
    ZX_design = np.concatenate((Z, X), axis=1)
    beta = np.random.randn(*(nvoxels, nparameters)).T
    p = pr.sigmoid(ZX_design @ beta)
    y = np.matrix(np.random.binomial(1, p))

    distbn = 'binomial'
    linkfn = 'logit'

    scores, _, _ = pr.compute_scores_mv(y, Z, X, distbn, linkfn)
"""
    betahat_0, fitted_values_0, psZ, _ = pr.glm_seq(
        y, Z, distbn, link, 'Fitting null glm model, progress:')
    glm0 = [betahat_0, fitted_values_0, psZ]

    #XZ_design = np.concatenate((Z, X), axis=1)
    # print('')
    # betahat_1, fitted_values_1, psX, _ = pr.glm_seq(
    #    y, XZ_design, distbn, link, 'Fitting full glm model, progress:')
    #glm1 = [betahat_1, fitted_values_1, psX]

    # Combine the perfect separations locations into a single vector
    pslocs = np.where(psZ)[0]
    #pslocs = np.unique(np.concatenate((np.where(psZ)[0], np.where(psX)[0])))

    # Get the important constants
    nvox = y.shape[1]
    nsubj = y.shape[0]
    p = X.shape[1]

    # Compute the scores
    print('\nComputing scores...')
    scores = np.zeros((nvox, p, nsubj))
    for i in np.setdiff1d(np.arange(nvox), pslocs):
        # print(np.sum(fitted_values_0[:, i]))
        # yvox = np.matrix(y[:, i]).T
        yvox = y[:, i]
        fitted_0_vox = np.matrix(fitted_values_0[:, i]).T
        scores[i, :, :] = pr.compute_scores(
            yvox, Z, X, fitted_0_vox, distbn, link, 'effective')

    return scores, pslocs, glm0


def compute_scores_from_glm(y, Z, X, pslocs, fitted_values_0, distbn, link, score_type='effective'):
    """
   Inputs:
     y: a numpy matrix of shape(n_samples, 1) representing the
            dependent variable
     Z: a numpy array of shape(n_samples, n_nuisance_parameters) representing
            the nuisance variables
     X: a numpy array of shape(n_samples, n_parameters) representing the
            independent variables
     pslocs:
     fitted_values_0:
     fitted_values_1:
     distbn: a string specifying the distribution of the response variable.
            Must be one of['binomial', 'gamma', 'gaussian', 'inverse gaussian',
                                               'negative binomial', 'poisson']
     linkfn: a string specifying the link function. Must be one of['log',
                'logit', 'probit', 'cauchy', 'cloglog', 'identity', 'inverse']
     score_type: 
    """
    # Get the important constants
    nvox = y.shape[1]
    nsubj = y.shape[0]
    p = X.shape[1]

    scores = np.zeros((nvox, p, nsubj))
    for i in np.setdiff1d(np.arange(nvox), pslocs):
        # print(np.sum(fitted_values_0[:, i]))
        # yvox = np.matrix(y[:, i]).T
        yvox = y[:, i]
        fitted_0_vox = np.matrix(fitted_values_0[:, i]).T
        scores[i, :, :] = pr.compute_scores(
            yvox, Z, X, fitted_0_vox, distbn, link, score_type)

    return scores


def compute_scores(y, Z, X, fitted_values_0, family, link, score_type='effective'):
    """
   Inputs:
     y: a numpy matrix of shape(n_samples, 1) representing the
            dependent variable
     Z: a numpy array of shape(n_samples, n_nuisance_parameters) representing
            the nuisance variables
     X: a numpy array of shape(n_samples, n_parameters) representing the
            independent variables
     fitted_values_0:
     fitted_values_1:
     family: a string specifying the distribution of the response variable.
            Must be one of['Binomial', 'Gamma', 'Gaussian', 'Inverse Gaussian',
                                               'Negative Binomial', 'Poisson']
     linkfn: a string specifying the link function. Must be one of['log',
                'logit', 'probit', 'cauchy', 'cloglog', 'identity', 'inverse']
    """
    # Doesn't seem like the fitted_values are being used, double check in own time and then fix this!
    nsubj = y.shape[0]

    # Obtain the derivative and variance evaluated at the fitted values
    Dhat0, Vhat0 = pr.get_par_expo_fam(fitted_values_0, family, link)

    # Initialize the weights matrix to be ones for now - this may change!
    W = Dhat0**2/Vhat0
    sqrtW = np.matrix(np.sqrt(W)).T

    # Compute the inverse square root of V
    sqrtinvVvect = Vhat0**(-0.5)

    # Obtain the residuals
    null_residuals = (y-fitted_values_0)

    sqrtinvVvect_times_residuals = np.multiply(sqrtinvVvect, null_residuals)

    if score_type == 'effective':
        A = np.multiply(Z.T, sqrtW)  # calculate Z transpose times diag(sqrtW)
        XTsqrtW = np.multiply(X.T, sqrtW)
        H = A.T @ np.linalg.inv(A @ A.T) @ A
        scores = np.multiply(XTsqrtW @ (np.identity(nsubj) - H),
                             sqrtinvVvect_times_residuals.T/(nsubj**0.5))
    return scores


def glm_seq(y, X, distbn, linkfn, progress_message='Progress:'):
    """ glm_seq runs the generalized linear model sequentially at each voxel in
     the array y.
    Inputs:

        y: a numpy array of shape(n_samples, n_voxels) representing the
            dependent variable
        X: a numpy array of shape(n_samples, n_parameters) representing the
            independent variables
        distbn: a string specifying the distribution of the response variable.
            Must be one of['Binomial', 'Gamma', 'Gaussian', 'Inverse Gaussian', 'Negative Binomial', 'Poisson']
        linkfn: a string specifying the link function. Must be one of['log',
                'logit', 'probit', 'cauchy', 'cloglog', 'identity', 'inverse']
    Output:
       betahat: a numpy array of shape(n_parameters, n_voxels) containing the
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
    gammahat, fitted_values, _, _ = pr.glm_seq(y, X, 'Binomial', 'logit')
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
    gammahat, fitted_values, _, _ = pr.glm_seq(y, X, 'Binomial', 'logit')
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
    gammahat, fitted_values, _, _ = pr.glm_seq(y, X, 'Binomial', 'logit')

    # Many voxel example
    nvoxels = 1000
    nsubj = 100
    nparameters = 5  # dimension of each observation
    # (minus 1 because we have to add the intercept)
    X = np.random.randn(nsubj, nparameters-1)
    intercept = np.ones(nsubj).reshape(nsubj, 1)
    X = np.concatenate((intercept, X), axis=1)
    gamma = np.random.randn(*(nvoxels, nparameters)).T
    p = pr.sigmoid(X @ gamma)
    y = np.random.binomial(1, p)
    gammahat, fitted_values, _, _ = pr.glm_seq(y, X, 'Binomial', 'logit')
"""
    # Obtain the sm link function
    link_fn = getattr(sm.families.links, linkfn)
    link = link_fn()

    family_fn = getattr(sm.families, distbn.capitalize())
    family = family_fn(link=link)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Calculate the number of voxels
    nvox = y.shape[1]

    # Initialize an empty array to store the estimated coefficients
    betahat = np.empty((X.shape[1], nvox))

    # Initialize an empty array to store the pvalues
    pvalues = np.empty((X.shape[1], nvox))

    # Initialize the log-likelihood function
    llf = np.empty((1, nvox))[0]

    # Initialize an empty array to store the fitted values
    fitted_values = np.empty((y.shape[0], nvox))

    # Initialize an empty array to store the confidence intervals
    conf_ints = np.empty((X.shape[1], nvox, 2))

    # Initialize a vector to capture the voxels where there is perfect separation
    perfect_separations = np.zeros((1, nvox))[0]

    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            message='overflow encountered in exp')

    # Loop over the columns of y
    for i in range(nvox):
        # Measure progress
        pr.progress_bar(i, nvox, progress_message)

        # Fit a GLM for the i-th column of y
        model = sm.GLM(y[:, i], X, family=family)
        try:
            result = model.fit()

            # Extract the coefficients and store them in betahat
            betahat[:, i] = result.params

            # Extract the coefficients and store them in betahat
            pvalues[:, i] = result.pvalues

            # Extract the coefficients and store them in betahat
            llf[i] = result.llf

            # Extract the fitted values and store them in fitted_values
            fitted_values[:, i] = result.fittedvalues

            # Store the confidence intervals
            conf_ints[:, i, :] = result.conf_int()
        except:
            perfect_separations[i] = 1

    warnings.resetwarnings()
    warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

    perfect_separations_second_loc = np.where(
        np.sum(fitted_values == 0, axis=0))[0]

    perfect_separations[perfect_separations_second_loc] = 1

    perfect_separations_third_loc = np.where(
        np.sum(fitted_values == 1, axis=0))[0]

    perfect_separations[perfect_separations_third_loc] = 1

    # perfect_separations = np.unique(perfect_separations)
    # perfect_separations.sort()

    return betahat, fitted_values, perfect_separations, pvalues


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
        i.e. g ^ (-1)(Xbetahat) where g is the link function
    grad:

    Examples
    -----------------
    # Single Voxel example
    nsubj = 1000  # number of observations
    nparameters = 5  # dimension of each observation
    # (minus 1 because we have to add the intercept)
    X = np.random.randn(nsubj, nparameters-1)
    intercept = np.ones(nsubj).reshape(nsubj, 1)
    X = np.concatenate((intercept, X), axis=1)
    beta = np.random.randn(nparameters)
    p = pr.sigmoid(X @ beta)
    y = np.random.binomial(1, p)
    pr.run_glm(y, X)

    # Multiple voxel example
    nvoxels = 2
    nsubj = 1000
    nparameters = 10  # dimension of each observation
    # (minus 1 because we have to add the intercept)
    X = np.random.randn(nsubj, nparameters-1)
    intercept = np.ones(nsubj).reshape(nsubj, 1)
    X = np.concatenate((intercept, X), axis=1)
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
        fitted_values(array-like): the fitted values for which to compute the
                                    derivative and variance
        family(str): the exponential family to use, can be 'gaussian',
                    'binomial', 'poisson', or 'gamma'
        link(str): the link function to use, can be 'identity', 'log', 'logit',
                    'cloglog', 'probit', or 'cauchit'

    Returns:
        tuple: a tuple containing Dhat: derivative of the link function and
        Vhat: the variance function at the fitted values.

    Examples
    """
    fitted_values = np.array(fitted_values)
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
    link_fn, link_fn_deriv, _ = get_linkfn(link)

    # Compute the derivative of the link at the fitted values
    Dhat = link_fn_deriv(link_fn(fitted_values))

    return Dhat, Vhat


def get_linkfn(link):
    if link == 'identity':
        def link_fn_inv(x): return x
        def link_fn_inv_deriv(x): return 1
        def link_fn(x): return x
    elif link == 'log':
        def link_fn_inv(x): return np.exp(x)
        def link_fn_inv_deriv(x): return np.exp(x)
        def link_fn(x): return np.log(x)
    elif link == 'logit':
        def link_fn_inv(x): return np.exp(x)/(1 + np.exp(x))
        def link_fn_inv_deriv(x): return np.exp(x)/(1+np.exp(x))**2
        # def link_fn_inv_deriv(x): return np.exp(x)/(1 + np.exp(x)) - \
        # np.exp(x)*np.exp(x)/(1 + np.exp(x))**2

        def link_fn(x): return np.log(x/(1-x))
    elif link == 'cloglog':
        def link_fn_inv(x): return 1 - np.exp(-np.exp(x))
        def link_fn_deriv(x): return np.exp(-np.exp(x))*np.exp(x)
        # link_fn =
    elif link == 'probit':
        def link_fn_inv(x): return np.exp(-x ^ 2/2)/np.sqrt(2*np.pi)
        # link_fn_deriv =
        # link_fn =
    elif link == 'cauchit':
        def link_fn_inv(x): return 1/(np.pi*(1+x**2))
        # link_fn_deriv =
        # link_inv =
    else:
        raise Exception('The link function must be one of the specified ones')

    return link_fn, link_fn_inv_deriv, link_fn_inv


def sigmoid(x): return 1 / (1 + np.exp(-x))


def fit_generalized_linear_model(formula, data, link, family):
    """
    Fits a generalized linear model with the given formula and data.

    Parameters:
    formula(str): a string specifying the model in the form of
                  "response ~ predictor1 + predictor2 + ... + predictorN"
    data(pandas DataFrame): a DataFrame containing the predictor and response variables
    link(statsmodels link): the link function to use in the model
    family(statsmodels family): the distribution family to use in the model

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
    return betahat, fitted_values, niter
    return betahat, fitted_values, niter
