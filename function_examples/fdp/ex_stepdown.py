"""
Testing the stepdown function
"""
import pyrft as pr
import numpy as np
import sanssouci as sa
from sklearn.utils import check_random_state

dim = (10,10)
nsubj = 100
fwhm = 8
lat_data = pr.statnoise(dim,nsubj,fwhm)

contrast_matrix = np.array([[1,-1,0],[0,1,-1]])
n_groups = contrast_matrix.shape[1]
rng = check_random_state(101)
categ = rng.choice(n_groups, nsubj, replace = True)
design_2use = pr.group_design(categ)
pi0 = 0.9
lat_data_with_signal, signal = pr.random_signal_locations(lat_data, categ, contrast_matrix, pi0)
minp_perm, orig_pvalues, pivotal_stats, bootstore = pr.boot_contrasts(lat_data_with_signal, design_2use, contrast_matrix, store_boots = 1)
m = np.prod(lat_data.masksize)*contrast_matrix.shape[0]
ntrue = int(np.round(pi0 * m))
nfalse = m - ntrue

# %% Bootstrap Lambdas - Even though it increases a lot the power doesn't increase
# by much!
alpha = 0.1
lambda_quant_boot = np.quantile(pivotal_stats, alpha)
lambda_quant_boot_sd, stepdownset = pr.step_down(bootstore, alpha, do_fwer = 0)

print(lambda_quant_boot)
print(lambda_quant_boot_sd)

t_func, _, _ = pr.t_ref('linear')

tfamilyeval = t_func(lambda_quant_boot, m, m)
tfamilyeval_sd = t_func(lambda_quant_boot_sd, m, m)

all_pvalues = np.ravel(orig_pvalues.field)
max_FP_bound = sa.max_fp(np.sort(all_pvalues), tfamilyeval)
min_TP_bound = m - max_FP_bound
print(min_TP_bound/nfalse)

max_FP_bound = sa.max_fp(np.sort(all_pvalues), tfamilyeval_sd)
min_TP_bound = m - max_FP_bound
print(min_TP_bound/nfalse)

# %% Parametric Lambdas
lambda_quant = alpha
hommel_value = pr.compute_hommel_value(np.ravel(orig_pvalues.field), alpha)
lambda_quant_sd = lambda_quant/(hommel_value/m)

print(lambda_quant)
print(lambda_quant_sd)

tfamilyeval = t_func(lambda_quant, m, m)
tfamilyeval_sd = t_func(lambda_quant_sd, m, m)
tfamilyeval_sd_old = t_func(alpha, hommel_value, hommel_value)

max_FP_bound = sa.max_fp(np.sort(all_pvalues), tfamilyeval)
min_TP_bound = m - max_FP_bound
print(min_TP_bound/nfalse)

max_FP_bound = sa.max_fp(np.sort(all_pvalues), tfamilyeval_sd)
min_TP_bound = m - max_FP_bound
print(min_TP_bound/nfalse)

max_FP_bound = sa.max_fp(np.sort(all_pvalues), tfamilyeval_sd_old)
min_TP_bound = m - max_FP_bound
print(min_TP_bound/nfalse)
# %%
lat_data_with_signal, signal = pr.random_signal_locations(lat_data, categ, contrast_matrix, pi0)
#minp_perm, orig_pvalues, pivotal_stats, bootstore = pr.boot_contrasts(lat_data_with_signal, design_2use, contrast_matrix, store_boots = 0)

from scipy.stats import t
#orig_tstats, _, _ = pr.contrast_tstats_noerrorchecking(lat_data, design_2use, contrast_matrix)
contrast_matrix, nsubj, n_params = pr.contrast_error_checking(lat_data_with_signal,design_2use,contrast_matrix)
orig_tstats, _, _ = pr.contrast_tstats_noerrorchecking(lat_data_with_signal, design_2use, contrast_matrix)
#orig_tstats, _, _ = pr.contrast_tstats_noerrorchecking(lat_data_with_signal, design_2use, contrast_matrix)


# Calculate the p-values
orig_pvalues_test = 2*(1 - t.cdf(abs(orig_tstats.field), nsubj-n_params))

print(orig_pvalues_test)
#print(orig_pvalues.field)


# %%
plt.imshow(np.mean(lat_data_with_signal.field[:,:, np.where(categ == 0)], axis = 2))