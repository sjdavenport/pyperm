# -*- coding: utf-8 -*-
"""
Testing the stats and auxiliary functions
"""
import numpy as np
import pyrft as pr

def test_mvtstat():
    """ Testing the mvtstat function """
    for i in np.arange(2):
        if i == 0:
            dim = (50,)
        elif i == 1:
            dim = (50,50)

        nsubj = 20
        overall_dim = dim + (nsubj,)
        noise = np.random.randn(*overall_dim)
        tstat, xbar, std_dev = pr.mvtstat(noise)
        assert tstat.shape == dim
        assert xbar.shape == dim
        assert std_dev.shape == dim
        assert np.sum(std_dev > 0) == np.prod(dim)

def test_contrast_tstats():
    """ Testing the contrast_tstats function """
    # Note that the function always runs contrast_error_checking and
    # constrast_tstats_noerrorchecking, so these functions are automatically tested
    # via running it
    # Need to include a 1D example
    nsubj = 30
    dim = (10,10)
    for i in np.arange(3):
        if i == 0:
            categ = np.zeros(nsubj)
            C = np.array(1)
        elif i == 1:
            categ = np.random.binomial(1, 0.4, size = nsubj)
            C = np.array((1,-1))
        elif i == 2:
            categ = np.random.multinomial(2, [1/3,1/3,1/3], size = nsubj)[:,1]
            C = np.array([[1,-1,0],[0,1,-1]]);

    X = pr.group_design(categ); lat_data = pr.wfield(dim,nsubj)
    tstat, residuals = pr.contrast_tstats(lat_data, X, C)

    assert isinstance(tstat, pr.classes.Field)
    assert tstat.D == len(dim)
    assert tstat.masksize == dim

    if len(C.shape) < 2:
        C = np.array([C])
    assert tstat.fieldsize == dim + (C.shape[0],)

    assert isinstance(residuals, np.ndarray)
    assert residuals.shape == dim + (nsubj,)

def test_fwhm2sigma():
    """ Testing the fwhm2sigma function """
    FWHM = 3
    sigma = pr.fwhm2sigma(FWHM)
    assert sigma > 0

def test_group_design():
    """ Testing the group_design function """
    for i in np.arange(2):
        if i == 0:
            categ = (0,1,1,0)
        elif i == 1:
            categ = (0,1,1,2,2)

        X = pr.group_design(categ)

        assert X.shape[0] == len(categ)
        assert X.shape[1] == len(np.unique(categ))
        assert np.sum(np.ravel(X)) == len(categ)

def test_tstat2pval():
    """ Testing the tstat2pval function """
    assert pr.tstat2pval(0, 10, one_sample = 0) == 1.0
    assert pr.tstat2pval(0, 10, one_sample = 1) == 0.5

    assert pr.tstat2pval(1.97, 1000) < 0.05
    assert pr.tstat2pval(1.96, 1000) > 0.05

    assert pr.tstat2pval(1.97, 1000, 1) < 0.025
    assert pr.tstat2pval(1.96, 1000, 1) > 0.025