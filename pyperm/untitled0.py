# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:59:14 2021

@author: ll17354fl = FirthLogisticRegression()
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import sys
import warnings
import math
import statsmodels
import numpy as np
from scipy import stats
import statsmodels.api as smf


def firth_likelihood(beta, logit):
    return -(logit.loglike(beta) + 0.5*np.log(np.linalg.det(-logit.hessian(beta))))


def null_fit_firth(y, X, start_vec=None, step_limit=1000, convergence_limit=0.0001):
    """
    Computes the null model in the likelihood ratio test 

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.  Make sure X has an intercept 
        term (column of ones).
    y : array-like of shape (n_samples,)
        Target vector relative to X. Please note this function only currently works for 
        binomial regression so output values of {0, 1} will work while 
        {0, 1, 2} will not. 
    start_vec : int or None, optional
        starting vector The default is None.
    step_limit : TYPE, optional
        Max number of steps before MLE termination. The default is 1000.
    convergence_limit : TYPE, optional
        Minimum difference between MLE's. The default is 0.0001.

    Returns
    -------
    return_fit :  
            intercept: Intercept coeffcient 
            beta: list of beta coeffcients  
            bse: coeffcient standard errors 
            fitll: fit log-likelihood
    """

    logit_model = smf.Logit(y, X)

    if start_vec is None:
        start_vec = np.zeros(X.shape[1])

    beta_iterations = []
    beta_iterations.append(start_vec)
    for i in range(0, step_limit):
        pi = logit_model.predict(beta_iterations[i])
        W = np.diagflat(np.multiply(pi, 1-pi))
        var_covar_mat = np.linalg.pinv(
            -logit_model.hessian(beta_iterations[i]))

        # build hat matrix
        rootW = np.sqrt(W)
        H = np.dot(np.transpose(X), np.transpose(rootW))
        H = np.matmul(var_covar_mat, H)
        H = np.matmul(np.dot(rootW, X), H)

        # penalised score
        U = np.matmul(np.transpose(X), y - pi +
                      np.multiply(np.diagonal(H), 0.5 - pi))
        new_beta = beta_iterations[i] + np.matmul(var_covar_mat, U)

        # step halving
        j = 0
        while firth_likelihood(new_beta, logit_model) > firth_likelihood(beta_iterations[i], logit_model):
            new_beta = beta_iterations[i] + 0.5*(new_beta - beta_iterations[i])
            j = j + 1
            if (j > step_limit):
                sys.stderr.write(
                    'Firth regression failed. Try increasing step limit.\n')
                return None

        beta_iterations.append(new_beta)
        if i > 0 and (np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) < convergence_limit):
            break

    return_fit = None
    if np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) >= convergence_limit:
        sys.stderr.write('Firth regression failed to converge.\n')
    else:
        # Calculate stats
        fitll = -firth_likelihood(beta_iterations[-1], logit_model)
        intercept = beta_iterations[-1][0]
        beta = beta_iterations[-1][1:].tolist()
        bse = np.sqrt(np.diagonal(
            np.linalg.pinv(-logit_model.hessian(beta_iterations[-1]))))

        return_fit = intercept, beta, bse, fitll

    return return_fit


class Firth_LogisticRegression(LogisticRegression,
                               ClassifierMixin,
                               BaseEstimator):
    """
    This class represents a rewriting Firth regression originally implemented 
    by John Lees (https://gist.github.com/johnlees/3e06380965f367e4894ea20fbae2b90d)
    into a class which can interact with the sci-kit learn ecosystem. 

    To use the fit function make sure X has an intercept term (column of ones).
    When using validation functions make sure to not include this 'dummy' column
    of ones.

    Please note: This estimator class does not currently pass the check_estimator test 
    in sklearn. This is because it cannot perform the multinomial classification task that
    check_estimator attempts to pass it.

    Parameters
        ----------

    start_vec : ndarray of shape (n_features, 1). Default set to None in which 
    case the zero vector is used.

    step_limit : int. 

    convergence_limit : float. 

    multi_class : string. Default is set to 'ovr' to let this function intgerate 
    with the logistic_regression parent class and pass the _check_multi_class
    function. A bit hacky but works.


    Antributes
        ----------

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `coef_` corresponds
        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).

    beta_ : list of size n_features. This is used in the wald and likelihood 
    ratio test functions.

    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.        

    """

    def __init__(self, start_vec=None,
                 step_limit=1000,
                 convergence_limit=0.0001,
                 multi_class='ovr'):
        self.start_vec = start_vec
        self.step_limit = step_limit
        self.convergence_limit = convergence_limit
        self.multi_class = multi_class  # multiclass should not be changed from 'ovr'

    def fit(self, X=None, y=None):
        """
        Fits the model accoridng to given training data. This fit function which 
        has been changed to work inaccordance with the sklearn estimator 
        documentation. Major changes are, rather than returning specific 
        variables fit() return an instance of itself allowing other functions 
        to be run from it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.  Make sure X has an intercept 
            term (column of ones).
        y : array-like of shape (n_samples,)
            Target vector relative to X. Please note this function only currently works for 
            binomial regression so output values of {0, 1} will work while 
            {0, 1, 2} will not. 

        Returns
        -------
        self
            Fitted estimator.

            self.fitll_ : fit log-likelihood
            self.intercept_ : intercept
            self.coef_ : coeffcients not including intercept (used in all other sklearn classes )
            self.beta_ : coeffcients including intercept (used in wald and LR tests)
            self.bse_ : standard errors

        """
        X, y = check_X_y(X, y)
        self.n_features_in = X.shape[1]-1
        self.classes_ = np.unique(y)

        logit_model = smf.Logit(y, X)

        if self.start_vec is None:
            start_vec = np.zeros(X.shape[1])

        beta_iterations = []
        beta_iterations.append(start_vec)
        for i in range(0, self.step_limit):
            pi = logit_model.predict(beta_iterations[i])
            W = np.diagflat(np.multiply(pi, 1-pi))
            var_covar_mat = np.linalg.pinv(
                -logit_model.hessian(beta_iterations[i]))

            # build hat matrix
            rootW = np.sqrt(W)
            H = np.dot(np.transpose(X), np.transpose(rootW))
            H = np.matmul(var_covar_mat, H)
            H = np.matmul(np.dot(rootW, X), H)

            # penalised score
            U = np.matmul(np.transpose(X), y - pi +
                          np.multiply(np.diagonal(H), 0.5 - pi))
            new_beta = beta_iterations[i] + np.matmul(var_covar_mat, U)

            # step halving
            j = 0
            while firth_likelihood(new_beta, logit_model) > firth_likelihood(beta_iterations[i], logit_model):
                new_beta = beta_iterations[i] + \
                    0.5*(new_beta - beta_iterations[i])
                j = j + 1
                if (j > self.step_limit):
                    sys.stderr.write(
                        'Firth regression failed. Try increasing step limit.\n')
                    return None

            beta_iterations.append(new_beta)
            if i > 0 and (np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) < self.convergence_limit):
                break

        if np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) >= self.convergence_limit:
            sys.stderr.write('Firth regression failed to converge\n')
        else:
            # Calculate stats
            self.fitll_ = -firth_likelihood(beta_iterations[-1], logit_model)
            self.intercept_ = beta_iterations[-1][0]
            # for other sklearn functions
            self.coef_ = np.array(
                beta_iterations[-1][1:].tolist()).reshape((1, self.n_features_in))
            # used by Wald and LR test
            self.beta_ = [self.intercept_] + beta_iterations[-1][1:].tolist()
            self.bse_ = np.sqrt(np.diagonal(
                np.linalg.pinv(-logit_model.hessian(beta_iterations[-1]))))

        return self

    def test_wald(self):
        '''
        Implemnatation of the wald test

        Returns
        -------
        waldp : list
            A list p-values from the Wald test.

        '''
        check_is_fitted(self)
        waldp = []
        for beta_val, bse_val in zip(self.beta_, self.bse_):
            waldp.append(2 * (1 - stats.norm.cdf(abs(beta_val/bse_val))))
        return waldp

    def test_likelihoodratio(self, X, y, start_vec=None, step_limit=1000, convergence_limit=0.0001):
        """
        Implementation of the likelihood ratio test. An external function, 
        null_fit_firth(), is used to refit the null-estimator.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features. Make sure to include the dummy column 
            of ones.
        y : array-like of shape (n_samples,)
            Target vector relative to X. 

        Returns
        -------
        lrtp : List
            List of p-values from the likelihood ratio test.

        """
        check_is_fitted(self)
        X_np = X.values
        lrtp = []
        for beta_idx, (beta_val, bse_val) in enumerate(zip(self.beta_, self.bse_)):
            null_X = np.delete(X_np, beta_idx, axis=1)
            (null_intercept, null_beta, null_bse, null_fitll) = null_fit_firth(
                y, null_X, start_vec, step_limit, convergence_limit)
            lrstat = -2*(null_fitll - self.fitll_)
            lrt_pvalue = 1
            if lrstat > 0:  # non-convergence
                lrt_pvalue = stats.chi2.sf(lrstat, 1)
            lrtp.append(lrt_pvalue)

        return lrtp
