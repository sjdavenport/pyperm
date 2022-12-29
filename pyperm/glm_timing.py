# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:43:51 2022

@author: 12SDa
"""

import time
import numpy as np 
start = time.time()
nvoxels = 40000
nsubj = 100
nparameters = 2 # dimension of each observation
X = np.random.randn(nsubj, nparameters) # (minus 1 because we have to add the intercept)
beta = np.random.randn(*(nvoxels, nparameters))
p = pr.sigmoid(X @ beta.T)
y = np.random.binomial(1, p)
betahat, fitted_values, grad = pr.run_glm(y, X, nruns = 100)
end = time.time()
print(end - start)