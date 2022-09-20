"""
Functions to simulate and test when the null is not true everywhere
"""
import pyperm as pr
import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\12SDa\\davenpor\\davenpor\\Other_Toolboxes\\sanssouci.python')
import sanssouci as sa
from sklearn.utils import check_random_state

def random_signal_locations(lat_data, categ, C, pi0, scale = 1, rng = check_random_state(101)):
    """ A function which generates random data with randomly located non-zero 
    contrasts

    Parameters
    -----------------
    lat_data: an object of class field
        giving the data with which to add the signal to
    categ: numpy.nd array,
        of shape (nsubj, 1) each entry identifying a category 
    C: numpy.nd array
        of shape (ncontrasts, nparams)
        contrast matrix
    pi0: a float 
        between 0 and 1 giving the proportion of the hypotheses that are truly null
    scale: float,
        giving the amount (scale of) the signal which to add. Default is 1.
    rng: random number generator

    Returns
    -----------------
    lat_data: an object of class field,
        which consists of the noise plus the added signal

    Examples
    -----------------
    dim = (25,25); nsubj = 300; fwhm = 4
    lat_data = pr.statnoise(dim, nsubj, fwhm)
    C = np.array([[1,-1,0],[0,1,-1]])
    n_groups = C.shape[1]
    rng = check_random_state(101)
    categ = rng.choice(n_groups, nsubj, replace = True)
    ld,sig = pr.random_signal_locations(lat_data, categ, C, 0.5)
    subjects_with_1s = np.where(categ==1)[0]
    plt.imshow(np.mean(ld.field[...,subjects_with_1s],2))
    subjects_with_2s = np.where(categ==2)[0]
    plt.imshow(np.mean(ld.field[...,subjects_with_2s],2))
    
    # Mean zero everywhere:
    ld,sig = pr.random_signal_locations(lat_data, categ, C, 1)

    """
    # Compute important constants
    dim = lat_data.masksize
    n_contrasts = C.shape[0]
    
    # Compute derived constants
    nvox = np.prod(dim) # compute the number of voxels
    m = nvox*n_contrasts # obtain the number of voxel-contrasts
    ntrue = int(np.round(pi0 * m)) # calculate the closest integer to make the 
                                #proportion of true null hypotheses equal to pi0
    
    # Initialize the true signal vector
    signal_entries = np.zeros(m)
    signal_entries[ntrue:] = 1
    
    if isinstance(dim, int):
        signal = pr.make_field(np.zeros((dim,n_contrasts)))
    else:
        signal = pr.make_field(np.zeros(dim + (n_contrasts,)))
            
    # Generate the signal by random shuffling the original signal
    # (if the proportion of signal is non-zero)
    if pi0 < 1:
        shuffle_idx = rng.choice(m, m, replace = False)
        shuffled_signal = signal_entries[shuffle_idx]
        spatial_signal2add = np.zeros(dim)
        for j in np.arange(n_contrasts):
            contrast_signal = shuffled_signal[j*nvox:(j+1)*nvox]
            signal.field[..., j] = contrast_signal.reshape(dim)
            spatial_signal2add += signal.field[..., j]
            subjects_with_this_contrast = np.where(categ==(j+1))[0]

            # Add the signal to the field
            for k in np.arange(len(subjects_with_this_contrast)):
                lat_data.field[..., subjects_with_this_contrast[k]] += scale*spatial_signal2add
    
    return lat_data, signal

def bootpower(dim, nsubj, contrast_matrix, fwhm = 0, design = 0, n_bootstraps = 100, niters = 1000, pi0 = 1, simtype = 1, alpha = 0.1, template = 'linear', replace = True, ):
    """ bootpower generates non-null simulations and calculates the proportion
    of the true alternatives that are identified

    Parameters
    -----------------
    dim:
    nsubj:
    contrast_matrix:
    fwhm:
    design:
    n_bootstraps:
    niters:
    pi0:
    alpha:
    template:
    replace:
    simtype:
        
    Returns
    -----------------
    
    Examples
    -----------------
    % Calculate the power using the bootstrap method
    dim = (4,4); nsubj = 100; C = np.array([[1,-1,0],[0,1,-1]]);
    power, power_sd = pr.bootpower(dim, nsubj, C, 4, 0, 100, 1000, 0.8)
    
    % Calculate the power using ARI
    dim = (25,25); nsubj = 10; C = np.array([[1,-1,0],[0,1,-1]]);
    power, power_sd = pr.bootpower(dim, nsubj, C, 4, 0, 100, 1000, 0.8, -1)
    """
    # Obtain ordered randomness
    rng = check_random_state(101)
    
    # If the proportion of null hypotheses has been set to 1, stop and return nothing
    if pi0 == 1:
        print('Power equals zero if pi0 = 1')
        return

    # If the design input is a matrix take this to be the design matrix
    # of the covariates (otherwise a random design is generated - see below)
    if not isinstance(design, int):
        design_2use = design

    # Obtain the inverse template function (allowing for direct input as well!)
    if isinstance(template, str):
        t_func, _, _ = pr.t_ref(template)

    if len(contrast_matrix.shape) == 1:
        n_contrasts = 1
        n_groups = 1
    else:
        n_contrasts = contrast_matrix.shape[0]
        n_groups = contrast_matrix.shape[1]
        
    # Initialize the true signal vector
    nvox = np.prod(dim)
    m = nvox*n_contrasts
    ntrue = int(np.round(pi0 * m))
    nfalse = m - ntrue
    
    # Initialize the vector to store the power
    power = np.zeros(5)
    power_sd = np.zeros(5)
    
     # Calculate TDP ratio for each bootstrap iteration
    for i in np.arange(niters):
        # Keep track of the progress.
        pr.modul(i,100)

        # Generate the data (i.e. generate stationary random fields)
        lat_data = pr.statnoise(dim,nsubj,fwhm)

        if isinstance(design, int):
            # Generate a random category vector with choices given by the design matrix
            categ = rng.choice(n_groups, nsubj, replace = True)

            # Ensure that all categories are present in the category vector
            while len(np.unique(categ)) < n_groups:
                print('had rep error')
                categ = rng.choice(n_groups, nsubj, replace = True)

            # Generate the corresponding design matrix
            design_2use = pr.group_design(categ)
        
        # Add random signal to the data
        lat_data, signal = pr.random_signal_locations(lat_data, categ, contrast_matrix, pi0, rng = rng)
                    
        # Convert the signal to boolean
        signal.field = signal.field == 0

        if simtype > -1:
            # Run the bootstrap algorithm
            if simtype == 1:
                # Implement the bootstrap algorithm on the generated data
                minp_perm, orig_pvalues, pivotal_stats, bootstore = pr.boot_contrasts(lat_data, 
                                                design_2use, contrast_matrix, n_bootstraps, template, 
                                                                        replace, store_boots = True)
            else:
                pr.perm_contrasts(lat_data, design_2use, contrast_matrix, n_bootstraps, template)
                
            # Calculate the lambda alpha level quantile for JER control
            lambda_quant = np.quantile(pivotal_stats, alpha)
            
            # Run the step down algorithm
            lambda_quant_sd, _ = pr.step_down( bootstore, alpha = alpha, 
                                              do_fwer = 0, template = template)
        
        # Only run this is pi0 < 1 as if it equals 1 then the power is zero
        if pi0 < 1:
            # Calulate the template family
            if simtype < 0:
                # Calculate the p-values
                orig_tstats, _, _ = pr.contrast_tstats_noerrorchecking(lat_data, design_2use, contrast_matrix)
                n_params = design_2use.shape[1]
                orig_pvalues = pr.tstat2pval(orig_tstats, nsubj - n_params)

                # Set the lambda quantile for simes
                lambda_quant = alpha

                # Run ARI
                hommel = pr.compute_hommel_value(np.ravel(orig_pvalues.field), alpha)
                lambda_quant_sd = lambda_quant/(hommel/m)

            # Calculate the template family at lambda_quant and lambda_quant_sd        
            tfamilyeval = t_func(lambda_quant, m, m)
            tfamilyeval_sd = t_func(lambda_quant_sd, m, m)

            # Update the power calculation
            power = pr.BNRpowercalculation_update(power, tfamilyeval, orig_pvalues, signal, m, nfalse);    
            power_sd = pr.BNRpowercalculation_update(power_sd, tfamilyeval_sd, orig_pvalues, signal, m, nfalse);

    # Calculate the power (when the data is non-null) by averaging over all simulations
    if pi0 < 1:
        power = power/niters
        power_sd = power_sd/niters

    return power, power_sd

def BNRpowercalculation_update(power, thr, orig_pvalues, signal, m, nfalse):
    # a) R = N_m
    all_pvalues = np.ravel(orig_pvalues.field)
    max_FP_bound = sa.max_fp(np.sort(all_pvalues), thr)
    min_TP_bound = m - max_FP_bound
    power[0] += min_TP_bound/nfalse
    
    # b) R_b denotes the rejection set that considers the voxel-contrasts
    # whose p-value is less than 0.05
    
    # Calculate the rejection set
    R_b = orig_pvalues.field < 0.05
    
    # Calculate the number of rejections of non-null hypotheses
    number_of_non_nulls = np.sum(R_b*signal.field > 0)
    
    # If there is at least 1 non-null rejection, record the TDP bound
    if number_of_non_nulls > 0.5:
        pval_set = np.sort(np.ravel(orig_pvalues.field[R_b]))
        npcount = len(pval_set)
        max_FP_bound_b = sa.max_fp(pval_set, thr)
        min_TP_bound_b = npcount - max_FP_bound_b
        power[1] += min_TP_bound_b/npcount
        
    # b) R_b denotes the rejection set that considers the voxel-contrasts
    # whose p-value is less than 0.05
    
    # Calculate the rejection set
    R_b = orig_pvalues.field < 0.1
    
    # Calculate the number of rejection of non-null hypotheses
    number_of_non_nulls = np.sum(R_b*signal.field > 0)
    
    # If there is at least 1 non-null rejection, record the TDP bound
    if number_of_non_nulls > 0.5:
        pval_set = np.sort(np.ravel(orig_pvalues.field[R_b]))
        npcount = len(pval_set)
        max_FP_bound_b = sa.max_fp(pval_set, thr)
        min_TP_bound_b = npcount - max_FP_bound_b
        power[2] += min_TP_bound_b/npcount
        
    # c) BH rejection set 0.05
    R_c, _, _ = pr.fdr_bh( all_pvalues, alpha = 0.05)
    number_of_non_nulls = np.sum(R_c*np.ravel(signal.field) > 0)
    
    # If there is at least 1 non-null rejection, record the TDP bound
    if number_of_non_nulls > 0.5:
        R_c_pvalues = all_pvalues[R_c]
        npcount = len(R_c_pvalues)
        max_FP_bound_c = sa.max_fp(np.sort(R_c_pvalues), thr)
        min_TP_bound_c = npcount - max_FP_bound_c
        power[3] += min_TP_bound_c/npcount
        
    # c) BH rejection set 0.1
    R_c, _, _ = pr.fdr_bh( all_pvalues, alpha = 0.1)
    number_of_non_nulls = np.sum(R_c*np.ravel(signal.field) > 0)
    
    # If there is at least 1 non-null rejection, record the TDP bound
    if number_of_non_nulls > 0.5:
        R_c_pvalues = all_pvalues[R_c]
        npcount = len(R_c_pvalues)
        max_FP_bound_c = sa.max_fp(np.sort(R_c_pvalues), thr)
        min_TP_bound_c = npcount - max_FP_bound_c
        power[4] += min_TP_bound_c/npcount
        
    return power

def simes_hommel_value(pvalues, alpha):
    """ simes_hommel_value computes the hommel value that arises from using
    the simes reference family at a level alpha given a set of p-values.

    Parameters
    -----------------
    pvalues: np.ndarray,
        giving the pvalues (each of which is greater than or equal to 0 and 
        less than or equal to 1)
    alpha: double,
        the level at which to control the error rate
        
    Returns
    -----------------
    hommel_value: int,
        the estimated hommel value
    
    Examples
    -----------------
    ## Example 1 (from Meijer 2018's Figure 1)
    
    pvalues = np.array((0, 0.01, 0.08, 0.1, 0.5, 0.7, 0.9))
    alpha = 0.251
    # Calculate the hommel value
    h = pr.simes_hommel_value(pvalues, alpha)
    
    # Plot the p-values and the critical slope
    sorted_pvalues = np.sort(pvalues)
    m = len(pvalues)
    plt.plot(np.arange(m)+1, sorted_pvalues, '*')
    critical_slope = (alpha - sorted_pvalues[m-h-1])/h
    xvals = np.arange(1, m+1)
    plt.plot(xvals, critical_slope*xvals + alpha - critical_slope*m)
    plt.ylim((0,1))
    print(h)
    
    ## Example 2 (used in the help for the hommel function in the hommel R package)
    pvalues = np.array((0.011432579, 0.009857651, 0.015673905, 0.010806814,
        0.001760031, 0.420629970, 0.976332934, 0.330314999, 0.892229400, 0.417906924))
    # Calculate the hommel value
    h = pr.simes_hommel_value(pvalues, 0.05)
    
    # Plot the p-values and the critical slope
    sorted_pvalues = np.sort(pvalues)
    m = len(pvalues)
    plt.plot(np.arange(m)+1, sorted_pvalues, '*')
    critical_slope = (alpha - sorted_pvalues[m-h-1])/h
    xvals = np.arange(1, m+1)
    plt.plot(xvals, critical_slope*xvals + alpha - critical_slope*m)
    plt.ylim((0,1))
    print(h)
    """
    # Ensure that 0 < alpha < 1
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha should be between 0 and 1')
        
    # Sort the pvalues
    pvalues = np.sort(pvalues)
    
    # Compute the number of hypotheses
    n_tests = len(pvalues)

    # If there is only one p-value p, the hommel value is 1 if p < alpha and 0 
    # otherwise.
    if len(pvalues) == 1:
        return pvalues[0] > alpha
    
    # If the smallest p-value is greater than alpha, return the number of tests
    # as the hommel value.
    if pvalues[0] > alpha:
        return n_tests
    
    if np.max(pvalues) < alpha:
        return 0
    
    # Compute the slopes between the points (i, p_i) and (n_tests, alpha)
    # note that the largest p-value is excluded because it would yield a slope
    # -infinity. This calculates (alpha - p_i)/(m - i) for i = 1, \dots, m-1.
    slopes = (alpha - pvalues[: - 1]) / np.arange(n_tests-1, 0, -1)
    
    # Find the index of the maximum slope (adding 1 to account for python indexing)
    slopes_argsort = np.argsort(slopes)
    max_slope_idx = slopes_argsort[n_tests-2] + 1
    
    # Compute the hommel value from this
    hommel_value = n_tests - (max_slope_idx)
    
    return hommel_value


def compute_hommel_value(p_vals, alpha):
    """Compute the All-Resolution Inference hommel-value
    Function taken from nilearn.
    
    Examples
    ----------------
    pvalues = np.array((0, 0.01, 0.08, 0.1, 0.5, 0.7, 0.9))
    m = len(pvalues)
    h = pr.compute_hommel_value(pvalues, 0.1)
    plt.plot(np.arange(m)+1, np.sort(pvalues), '*')
    print(h)
    
    pvalues = np.array((0.011432579, 0.009857651, 0.015673905, 0.010806814,
        0.001760031, 0.420629970, 0.976332934, 0.330314999, 0.892229400, 0.417906924))
    m = len(pvalues)
    h = pr.compute_hommel_value(pvalues, 0.1)
    plt.plot(np.arange(m)+1, np.sort(pvalues), '*')
    print(h)
    """
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha should be between 0 and 1')
    p_vals = np.sort(p_vals)
    n_tests = len(p_vals)

    if len(p_vals) == 1:
        return p_vals[0] > alpha
    if p_vals[0] > alpha:
        return n_tests
    slopes = (alpha - p_vals[: - 1]) / np.arange(n_tests -1, 0, -1)
    slope = np.max(slopes)
    hommel_value = np.trunc(alpha / slope)
    return np.minimum(hommel_value, n_tests)