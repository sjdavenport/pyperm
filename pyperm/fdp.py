"""
Functions to control the fdr
"""
import numpy as np
import pyperm as pr
import sanssouci as sa


def fdr_bh(pvalues, alpha=0.05):
    """ fdr_bh( pvalues, alpha ) implements the Benjamini-Hochberg procedure on
    a numpy array of pvalues, controlling the FDR to a level alpha

    Parameters
    ----------
    pvalues: numpy.Ndarray,
             a vector of p-values
    alpha: float,
           the significance level, default is 0.05

    Returns
    -------
    rejection_ind: boolean numpy array,
                   with the same size as pvalues such that a
                   given entry is 1 if that point is rejected and 0 otherwise
    n_rejections: int,
                  the total number of rejections
    rejection_locs:  int numpy array
                     the locations of the rejections

    Examples
    --------
    from scipy.stats import norm
    nvals = 100
    normal_rvs = np.random.randn(1, 100)[0]
    normal_rvs[0: 20] += 2
    pvalues = 1 - norm.cdf(normal_rvs)
    rejection_ind, n_rejections, sig_locs = pr.fdr_bh(pvalues)
    print(sig_locs)
    """
    # Get the dimension of the pvalues
    dim = pvalues.shape

    # Need the sort index for python!
    pvalues_vector = np.ravel(pvalues)
    sort_index = np.argsort(pvalues_vector)
    sorted_pvalues = pvalues_vector[sort_index]
    n_pvals = len(pvalues_vector)

    bh_upper = (np.arange(n_pvals) + 1) * alpha / n_pvals

    bh_vector = sorted_pvalues <= bh_upper

    # Find the indices j for which p_(j) \leq alpha*j/npvals
    # Note need to add + 1 to account for python indexing
    find_lessthans = np.where(bh_vector)[0] + 1

    if find_lessthans.shape[0] == 0:
        # If there are no rejections
        n_rejections = 0
        rejection_locs = np.zeros(0)
        rejection_ind = np.zeros(dim)
    else:
        # If there are rejections, find and store them and return them as output

        # Calculate the number of rejections
        n_rejections = find_lessthans[-1]

        # Obtain the rejections location indices
        rejection_locs = np.sort(sort_index[0: n_rejections])

        # Initialize a boolean vector of location of rejections
        rejection_ind = np.zeros(n_pvals, dtype='bool')

        # Set each rejection to 1
        rejection_ind[rejection_locs] = 1

        # Reshape rejection_ind so that it has the same size as pvalues
        rejection_ind = rejection_ind.reshape(dim)

    return rejection_ind, n_rejections, rejection_locs


def step_down(permuted_pvals, alpha=0.1, max_number_of_steps=50, do_fwer=1, template='linear'):
    """ step_down implements the general step down algorithm and the step down
    algorithm from Blanchard et al 2020 and Davenport et al 2022.

    Parameters
    ----------
    permuted_pvals: np.ndarry,
        of size (B,m) where B is the number of bootstraps/permutations and m is
        the number of hypotheses. This contains the permuted p-values from
        applying the bootstrap or permutation JER control methods (e.g. via
        pr.boot_contrasts). Importantly the first row of this matrix must
        correspond to the original p-values, and the remainder to permuted
        p-values. As the first permutation is always taken to be the original
        data.
    alpha:   float,
        the alpha level at which to control the false positive rate. Typically
        either taken to be 0.05 or 0.1 (but of course other values are just as
        reasonable).
    max_number_of_steps:   int,
        giving the maximum number of iterations you're willing to go through
        before terminating
    do_fwer:  bool,
        determining whether to do the joint step down procedure or just the
        fwer step down procedure. Default is the fwer one, i.e. True!
    template:   char,
        a character array specifying the type of template to use. Default is
        'linear', i.e. yielding the linear template.

    Returns
    -------
    alpha_quantile: float,
        the alpha quantile of the minimum of the pvalues over the step downed
        set in the fwer case OR
        the alpha quantile of the pivotal statistics in the jer case
    stepdownset: np.ndarray
        a vector giving the indices of the elements of the step down set!

    Examples
    --------

    # With signal
    lat_data = pr.statnoise(dim,nsubj,fwhm)
    lat_data, signal = pr.random_signal_locations(
        lat_data, categ, contrast_matrix, pi0, rng = rng)

    # With no signal
    dim = (10,10); N = 30; categ = np.random.multinomial(
        2, [1/3,1/3,1/3], size = N)[:,1]
    X = pr.group_design(categ); C = np.array(
        [[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(dim,N)
    minP, orig_pvalues, pivotal_stats, bootstore = pr.boot_contrasts(
        lat_data, X, C, store_boots = 1)
    lambda_quant, stepdownset = pr.step_down( bootstore )
    """
    # Calculate the number of bootstraps and the number of p-values
    B, m = permuted_pvals.shape

    # Obtain the template functions
    t_func, _, _ = pr.t_ref(template)

    # Initialize the set to record which pvalues are recorded as a boolean
    # variable
    store_stepdownset = np.ones(m, dtype='bool')

    # Initialize a vector to record the step down ste in terms of the original
    # indicies
    stepdownset = np.arange(m)

    # Initialize the number of steps through the algorithm
    no_of_steps = 0

    # Initialize the indicator that the step down sets match
    sets_match = 0

    while (sets_match == 0) and (no_of_steps < max_number_of_steps):
        if do_fwer:  # FWER STEP DOWN
            # Calculate the minimum p-values (one for each bootstrap i.e. computing
            # the minimum spatially)
            minpermutedpvals = np.min(permuted_pvals, axis=1)

            # Obtain the alpha quantile
            alpha_quantile = np.quantile(minpermutedpvals, alpha)

            # set the threshold which in this case equals the alpha quantile
            threshold = alpha_quantile
        else:  # JER STEP DOWN
            # Calculate the bootstrapped pivotal statistics
            # Note that the denominator in the template is always m
            pivotal_stats = pr.get_pivotal_stats(permuted_pvals, m, template)

            # Obtain the lambda quantile of the pivotal statistics
            alpha_quantile = np.quantile(pivotal_stats, alpha)

            # Obtain the threshold used for performing the step down algorithm
            threshold = t_func(alpha_quantile, 1, m)

        # threshold
        newstepdownset = permuted_pvals[0, :] > threshold

        # Update the stepdown storage set
        stepdownset = stepdownset[newstepdownset]

        # Test to see whether the set has changed (or not)
        if sum(store_stepdownset) == sum(newstepdownset):
            sets_match = 1
        else:
            # Update the set of pvalues to use
            store_stepdownset = newstepdownset

            # Restrict the original and permuted pvalue to the hypotheses
            # corresponding to the step down
            permuted_pvals = permuted_pvals[:, store_stepdownset]

            # Iterate the number of steps
            no_of_steps += 1

    return alpha_quantile, stepdownset


def get_bounds(pvalue_subset, lambda_quant, nhypotheses):
    """ get_bounds( pvalues, alpha )

    Parameters
    ----------
    pvalue_subset  np.ndarry,
        of a subset of pvalues on which TDP bounds are desired
    lambda_quant,  float
        the lambda quantile of the pivotal statistics
    nhypotheses, int
        the number of total hypotheses being tested

    Returns
    -------
    bounds

    Examples
    --------

    """
    thr = sa.linear_template(lambda_quant, nhypotheses, nhypotheses)

    npvals = len(pvalue_subset)

    FP_bound = sa.max_fp(pvalue_subset, thr)
    TP_bound = npvals - FP_bound

    FDP_bound = FP_bound/npvals
    TDP_bound = TP_bound/npvals

    return FP_bound, TP_bound, FDP_bound, TDP_bound
