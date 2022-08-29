# -*- coding: utf-8 -*-
"""
Testing the functions used to control the FDP/FDR
"""
import numpy as np
import pyrft as pr
from scipy.stats import norm

def test_fdr_bh():
    """ Testing the fdr_bh function """
    np.random.seed(10)
    nvals = 100
    normal_rvs = np.random.randn(1, nvals)[0]
    normal_rvs[0:20] += 2
    pvalues = 1 - norm.cdf(normal_rvs)
    rejection_ind, n_rejections, sig_locs = pr.fdr_bh(pvalues)

    assert isinstance(rejection_ind, np.ndarray)
    assert isinstance(sig_locs, np.ndarray)

    assert rejection_ind.shape == (nvals,)
    assert n_rejections == int(n_rejections)
    assert sig_locs.shape == (n_rejections,)
