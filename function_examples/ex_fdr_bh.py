# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from scipy.stats import norm
import pyrft as pr

niters = 10000; nvox = 100; ntrue = 0

fdr_est = 0;
for I in np.arange(niters):
    pr.modul(I, 1000)
    normal_rvs = np.random.randn(1,nvox)[0]
    normal_rvs[0:ntrue] = normal_rvs[0:ntrue] + 2
    pvalues = 1 - norm.cdf(normal_rvs)

    sig_locs, n_rejections, _ = pr.fdr_bh(pvalues)
    
    if n_rejections > 0:
       # print(sig_locs)
       fdr_est = fdr_est + np.sum( sig_locs[ntrue:] > 0 )/n_rejections

print(fdr_est/niters)
