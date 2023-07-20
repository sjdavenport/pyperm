# -*- coding: utf-8 -*-
"""
A file contain statistics functions
"""
# Import statements
import fnmatch
import os
from time import sleep
import sys
import numpy as np
import pyperm as pr
from scipy.stats import t, norm


def bernstd(p, nsubj, confidence_level=0.95):
    """ A function to compute the multivariate t-statistic

    Parameters
    -----------------
    p:  float,
        the value at which to generate confidence intervals about
    nsubj:  int,
        the number of subjects
    confidence_level: float,
        a number between 0 and 1 that gives the size of the confidence interval
        that is desired. Default is 0.95, i.e. yielding a 95% confidence interval

    Returns
    -----------------
    interval:  tuple,
        giving the left and right bounds of the confidence interval
    std_error: float,
        the standard error (i..e sigmahat)

    Examples
    -----------------
    pr.bernstd(0.05,1000,0.95)
    """
    # Calulate the standard error
    std_error = (p * (1 - p))**(1/2) * norm.ppf(1 -
                                                (1 - confidence_level)/2) / np.sqrt(nsubj)

    # Use this to generate a confidence interval
    interval = (p - std_error, p + std_error)

    return interval, std_error


def mvtstat(data):
    """ A function to compute the multivariate t-statistic

    Parameters
    -----------------
    data:  numpy.ndarray of shape (Dim, nsubj)
          Here Dim is the size of the field and nsubj is the number of subjects

    Returns
    -----------------
    tstat:   numpy.ndarray of shape (Dim)
          Each entry is the is the t-statistic calulcated across subjects
    mean:    numpy.ndarray of shape (Dim)
          Each entry is the is the mean calulcated across subjects
    std:     numpy.ndarray of shape (Dim)
          Each entry is the is the standard deviation calulcated across subjects

    Examples
    -----------------
    # tstat of random noise
    noise = np.random.randn(50,50,20); arrays = mvtstat(noise);  tstat = arrays[0]
    # For comparison to MATLAB
    a = np.arange(12).reshape((3,4)).transpose()+1; tstat = mvtstat(a)[0]
    """
    # Obtain the size of the array
    s_data = np.shape(data)

    # Obtain the dimensions
    dim = s_data[0:-1]

    # Obtain the number of dimensions
    n_dim = len(dim)

    # Obtain the number of subjects
    nsubj = s_data[-1]

    # Get the mean and stanard deviation along the number of subjects
    # Remember in python D is the last dimension of a D+1 array
    xbar = data.mean(n_dim)

    # Calculate the standard deviation (multiplying to ensure the population std is used!)
    std_dev = data.std(n_dim)*np.sqrt((nsubj/(nsubj-1.)))

    # Compute Cohen's d
    cohensd = xbar/std_dev
    tstat = np.sqrt(nsubj)*cohensd

    return tstat, xbar, std_dev


def contrast_tstats(lat_data, design, contrast_matrix, check_error=1):
    """ A function to compute the voxelwise t-statistics for a set of contrasts
    Parameters
    -----------------
    lat_data:  a numpy.ndarray of shape (Dim, N) or an object of class field
          giving the data where Dim is the spatial dimension and N is the number of subjects
          if a field then the fibersize must be 1 and the final dimension must be
          the number of subjects
    design: a numpy.ndarray of size (N,p)
        giving the covariates (p being the number of parameters)
    contrast_matrix: a numpy.ndarray of size (L,p)
        corresponding to the contrast matrix, such that which each row is a
        contrast vector (where L is the number of constrasts)
    check_error:  Bool,
          determining whether to perform error checking or not  (not always
          necessary e.g. during a permutation loop etc) default  is 1 i.e. to
          perform error checking

    Returns
    -----------------
    tstat_field: an object of class field
          which has spatial size the same as input data and fibersize equal
          to the number of contrasts
    residuals: a numpy array
        of shape (dim, nsubj) containing the residuals i.e. residuals(..., I)
        provides the imagewise residuals for the Ith subject

    Examples
    -----------------
    # One Sample tstat
    Dim = (3,3); N = 30; categ = np.zeros(N)
    X = pr.group_design(categ); C = np.array(1); lat_data = pr.wfield(Dim,N)
    tstat, residuals = pr.contrast_tstats(lat_data, X, C)

    # Compare to mvtstat:
    print(tstat.field.reshape(lat_data.masksize)); print(mvtstat(lat_data.field)[0])

    # Two Sample tstat
    Dim = (10,10); N = 30; categ = np.random.binomial(1, 0.4, size = N)
    X = pr.group_design(categ); C = np.array((1,-1)); lat_data = pr.wfield(Dim,N)
    tstats = pr.contrast_tstats(lat_data, X, C)

    # 3 Sample tstat (lol)
    Dim = (10,10); N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
    X = pr.group_design(categ); C = np.array([[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(Dim,N)
    tstats = pr.contrast_tstats(lat_data, X, C)
    """
    # Error check the inputs
    if check_error == 1:
        contrast_matrix, _, _ = contrast_error_checking(
            lat_data, design, contrast_matrix)

    # Convert the data to be a field if it is not one already
    if isinstance(lat_data, np.ndarray):
        lat_data = pr.make_field(lat_data)

    # Having now run error checking calculate the contrast t-statistics
    tstat_field, residuals, _ = contrast_tstats_noerrorchecking(
        lat_data, design, contrast_matrix)

    return tstat_field, residuals


def contrast_error_checking(lat_data, design, contrast_matrix):
    """ A function which performs error checking on the contrast data to ensure
    that it has the right dimensions.

        Parameters
    -----------------
    lat_data:  a numpy.ndarray of shape (Dim, N) or an object of class field
          giving the data where Dim is the spatial dimension and N is the number 
          of subjects if a field then the fibersize must be 1 and the final 
          dimension must be the number of subjects
    design: a numpy.ndarray of size (N,p)
        giving the covariates (p being the number of parameters)
    contrast_matrix: a numpy.ndarray of size (L,p)
        corresponding to the contrast matrix, such that which each row is a
        contrast vector (where L is the number of constrasts)

    Returns
    -----------------
    contrast_matrix: a numpy.ndarray of size (L,p)
        corresponding to the contrast matrix, such that which each row is a
        contrast vector (where L is the number of constrasts)
    nsubj: an int
        giving the number of subjects
    n_params: an int.
        giving the number of parameters in the model
    """
    # Error Checking
    # Ensure that C is a numpy array
    if not isinstance(contrast_matrix, np.ndarray):
        raise Exception("C must be a numpy array")

    # Ensure that C is a numpy matrix
    if len(contrast_matrix.shape) == 0:
        contrast_matrix = np.array([[contrast_matrix]])
    elif len(contrast_matrix.shape) == 1:
        contrast_matrix = np.array([contrast_matrix])
    elif len(contrast_matrix.shape) > 2:
        raise Exception("C must be a matrix not a larger array")

    # Calculate the number of parameters in C
    n_contrast_params = contrast_matrix.shape[1]  # parameters

    # Calculate the number of parameters p and subjects N
    nsubj = design.shape[0]  # subjects
    n_params = design.shape[1]  # parameters

    # Ensure that the dimensions of X and C are compatible
    if n_params != n_contrast_params:
        raise Exception(
            'The dimensions of design and contrast_matrix do not match')

    # Ensure that the dimensions of X and lat_data are compatible
    if nsubj != lat_data.fibersize:
        raise Exception(
            'The number of subjects in design and lat_data do not match')

    return contrast_matrix, nsubj, n_params


def contrast_tstats_noerrorchecking(lat_data, design, contrast_matrix):
    """ A function to compute the voxelwise t-statistics for a set of contrasts
    but with no error checking! For input into permutation so you do not have to
    run the error checking every time.

    Parameters
    -----------------
    lat_data:  an object of class field
          the data for N subjects on which to calculate the contrasts
    design: a numpy.ndarray of size (N,p)
        giving the covariates (p being the number of parameters)
    contrast_matrix: a numpy.ndarray of size (L,p)
        corresponding to the contrast matrix, such that which each row is a
        contrast vector (where L is the number of constrasts)

    Returns
    -----------------
    tstat_field: an object of class field
          which has spatial size the same as input data and fibersize equal
          to the number of contrasts
    residuals: a numpy array
        of shape (dim, nsubj) containing the residuals i.e. residuals(..., I)
        provides the imagewise residuals for the Ith subject
    Cbeta_field: an object of class field
        which has shape (dim, ncontrasts), for each contrast this gives the 
        c^Tbetahat image.

    Examples
    -----------------
    # One Sample tstat
    Dim = (3,3); N = 30; categ = np.zeros(N)
    X = pr.group_design(categ); C = np.array([[1]]); lat_data = pr.wfield(Dim,N)
    tstat, residuals, Cbeta_field = pr.contrast_tstats_noerrorchecking(lat_data, X, C)
    # Compare to mvtstat:
    print(tstat.field.reshape(lat_data.masksize)); print(mvtstat(lat_data.field)[0])

    # Two Sample tstat
    Dim = (10,10); N = 30; categ = np.random.binomial(1, 0.4, size = N)
    X = group_design(categ); C = np.array([[1,-1]]); lat_data = pr.wfield(Dim,N)
    tstat, residuals, Cbeta_field = pr.contrast_tstats_noerrorchecking(lat_data, X, C)

    # 3 Sample tstat (lol)
    Dim = (10,10); N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
    X = group_design(categ); C = np.array([[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(Dim,N)
    tstat, residuals, Cbeta_field = pr.contrast_tstats_noerrorchecking(lat_data, X, C)
    """
    # Calculate the number of contrasts
    n_contrasts = contrast_matrix.shape[0]  # constrasts

    # Calculate the number of parameters p and subjects N
    design = np.array(design)  # Ensure that the design matrix is an array
    nsubj = design.shape[0]  # subjects
    n_params = design.shape[1]  # parameters

    # rfmate = np.identity(p) - np.dot(X, np.dot(np.linalg.inv(np.dot(np.transpose(X), X)),z
    # np.transpose(X)))
    # Calculate (X^TX)^(-1)
    xtx_inv = np.linalg.inv(design.T @ design)

    # Calculate betahat (note leave the extra shaped 1 in as will be remultipling
    # with the contrast vectors!)
    betahat = xtx_inv @ design.T @ lat_data.field.reshape(
        lat_data.fieldsize + (1,))

    # Calculate the residual forming matrix
    rfmate = np.identity(nsubj) - design @ xtx_inv @ design.T

    # Compute the estimate of the variance via the residuals (I-P)Y
    # Uses a trick adding (1,) so that multiplication is along the last column!
    # Note no need to reshape back yet not doing so will be useful when dividing by the std
    residuals = rfmate @ lat_data.field.reshape(lat_data.fieldsize + (1,))

    # Square and sum over subjects to calculate the variance
    # This assumes that X has rank p!
    std_est = (np.sum(residuals**2, lat_data.D)/(nsubj-n_params))**(1/2)

    # Compute the t-statistics
    if lat_data.D == 1:
        Cbeta = (contrast_matrix @
                 betahat).reshape((lat_data.masksize[1], n_contrasts))
        tstats = Cbeta/std_est
    else:
        Cbeta = (contrast_matrix @
                 betahat).reshape(lat_data.masksize + (n_contrasts,))
        tstats = Cbeta/std_est

    # Scale by the scaling constants to ensure var 1
    for l in np.arange(n_contrasts):
        scaling_constant = (
            contrast_matrix[l, :] @ xtx_inv @ contrast_matrix[l, :])**(1/2)
        tstats[..., l] = tstats[..., l]/scaling_constant

    # Generate the field of tstats
    tstat_field = pr.Field(tstats, lat_data.mask)
    Cbeta_field = pr.Field(Cbeta, lat_data.mask)

    # Reshape the residuals back to get rid of the trailing dimension
    residuals = residuals.reshape(lat_data.fieldsize)

    return tstat_field, residuals, Cbeta_field


def fwhm2sigma(fwhm):
    """ A function translate the standard deviation to FWHM

    Parameters
    -----------------
    FWHM:    double,
          a value specifying the full width half max

    Returns
    -----------------
    sigma:    double,
          the sigma corresponding to the FWHM

    Examples
    -----------------
    # FWHM = 3; sigma = fwhm2sigma(FWHM)
    """
    sigma = fwhm / np.sqrt(8 * np.log(2))

    return sigma


def group_design(categ):
    """ A function to compute the covariate matrix for a given set of categories

    Parameters
    ------------------
    categ:  a tuple of integers of length N
        where N is the number of subjects). Each entry is number of the category
        that a given subject belongs to (enumerated from 0 to ncateg - 1)
        E.g: (0,1,1,0) corresponds to 4 subjects, 2 categories and
                 (0,1,2,3,3,2) corresponds to 6 subjects and 4 categories
            Could make a category class!

    Returns
    ------------------
    design: a design matrix that can be used to assign the correct categories

    Examples
    ------------------
    categ = (0,1,1,0); pr.group_design(categ)
    """
    # Calculate the number of subjects
    nsubj = len(categ)

    # Calculate the number of parameters i.e. the number of distinct groups
    n_params = len(np.unique(categ))

    # Ensure that the number of categories is not too high!
    if np.max(categ) > n_params - 1:
        raise Exception("the maximum category number should not exceed \
                    one minus the number of categories")

    # Initialize the design matrix
    design = np.zeros((nsubj, n_params))

    # Set the elements of the design matrix by assigning each subject a category
    for i in np.arange(nsubj):
        # change so you do this all at once if possible!
        design[i, int(categ[i])] = 1

    return design


def modul(iterand, niterand=100):
    """ A function which allows you to easily check how a for loop is
    % progressing by displaying iterand iff it is evenly divided by niterand

    Parameters
    ------------------
    iterand:
    niterand:

    Returns
    ------------------
    Prints iterand if niterand divides into iterand

    Examples
    ------------------
    pr.modul(100,10)
    pr.modul(3,5)
    """
    if iterand % niterand == 0:
        print(iterand)


def tstat2pval(tstats, df, one_sample=0):
    """ A function converts the test-statistics to pvalues

    Parameters
    ------------------
    tstats
    df:   int,
          the degrees of freedom of the t-statistic
    one_sample

    Returns
    ------------------
    pvalues:

    Examples
    ------------------
    zvals = np.random.randn(1, 10000)
    pvals = pr.tstat2pval( zvals[0], 1000, one_sample = 0 )
    plt.hist(pvals)

    whitenoise = pr.wfield((10,10), 2)
    pvals = tstat2pval( whitenoise, 1000)
    plt.hist(np.ravel(pvals.field))
    """
    if one_sample == 0:
        if isinstance(tstats, pr.classes.Field):
            pvalues = tstats
            pvalues.field = 2 * (1 - t.cdf(abs(tstats.field), df))
        else:
            pvalues = 2 * (1 - t.cdf(abs(tstats), df))
    else:
        if isinstance(tstats, pr.classes.Field):
            pvalues = tstats
            pvalues.field = 1 - t.cdf(tstats.field, df)
        else:
            pvalues = 1 - t.cdf(tstats, df)

    return pvalues


def loader(I, totalI):
    progress = 100*I/totalI
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" %
                     ('='*int(np.floor(progress/5)), progress))
    sys.stdout.flush()


def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='█'):
    """
    Call this function in a loop to create a progress bar in the console.
    :param iteration: current iteration (Int)
    :param total: total iterations (Int)
    :param prefix: prefix string (Str)
    :param suffix: suffix string (Str)
    :param decimals: positive number of decimals in percent complete (Int)
    :param length: character length of bar (Int)
    :param fill: bar fill character (Str)

    Examples:
    for i in range(total):
        time.sleep(0.01)  # Simulate a long computation
        progress_bar(i + 1, total, prefix='Progress:', suffix='Complete', length=50)

    Author: Chatgpt
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')


def combine_arrays(arrays):
    non_zero_arrays = [array for array in arrays if array.ndim > 0]
    if len(non_zero_arrays) == 0:
        return np.array([])
    else:
        return np.concatenate(non_zero_arrays)


def list_files(directory, search_string, add_dir=0):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if fnmatch.fnmatch(file_name, f'*{search_string}*'):
                if add_dir:
                    file_list.append(directory+file_name)
                else:
                    file_list.append(file_name)
    return file_list
