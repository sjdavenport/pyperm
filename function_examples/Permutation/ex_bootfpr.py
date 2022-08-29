import pyrft as pr
import numpy as np

# %% 1D no signal
dim = 5; nsubj = 30; C = np.array([[1,-1,0],[0,1,-1]]);
FWER_FPR, JER_FPR = pr.bootfpr(dim, nsubj, C)

# %% 1D with signal

# %% 2D - bootstrapping
dim = (10,10); nsubj = 100; C = np.array([[1,-1,0],[0,1,-1]]);
FWER_FPR, JER_FPR, FWER_FPR_SD, JER_FPR_SD = pr.bootfpr(dim, nsubj, C, pi0 = 0.5)

# %% 2D - Parametric
dim = (10,10); nsubj = 100; C = np.array([[1,-1,0],[0,1,-1]]);
Simes_FPR, Simes_FPR, ARI_FPR, ARI_FPR = pr.bootfpr(dim, nsubj, C, simtype = -1, pi0 = 0.5)

# %% 2D - ARI - no smoothing
dim = (10,10); nsubj = 100; C = np.array([[1,-1,0],[0,1,-1]]);
FWER_FPR, JER_FPR, _, _  = pr.bootfpr(dim, nsubj, C, simtype = -1)

# %% 2D - ARI - with smoothing
dim = (10,10); nsubj = 100; C = np.array([[1,-1,0],[0,1,-1]]);
FWER_FPR, JER_FPR, _, _  = pr.bootfpr(dim, nsubj, C, fwhm = 8, simtype = -1)

# %% 2D with signal
dim = (10,10); nsubj = 100; C = np.array([[1,-1,0],[0,1,-1]]);
fwhm = 0
FWER_FPR, JER_FPR, FWER_FPR_SD, JER_FPR_SD = pr.bootfpr(dim, nsubj, C, fwhm, 0, 100, 1000, 0.8)

# %% 2D with signal - ARI
dim = (10,10); nsubj = 100; C = np.array([[1,-1,0],[0,1,-1]]);
fwhm = 0
FWER_FPR, JER_FPR, _, _ = pr.bootfpr(dim, nsubj, C, fwhm, 0, 100, 1000, 0.8, simtype = -1)