"""
Functions to run cluster inference - test
"""
import math
import numpy as np
from skimage import measure
from nilearn.image import get_data, load_img
from nilearn.input_data import NiftiMasker
from scipy.stats import t
import sanssouci as sa
import pyperm as pr


def find_clusters(test_statistic, cdt, below=bool(0), mask=math.nan,
                  connectivity=1, two_sample=bool(0), min_cluster_size=1):
    """ find_clusters
    Parameters
    ---------------------
    test_statistic:   a numpy.nd array,
    cdt:    a double,
        the cluster defining threshold
    below: bool,
       whether to define the clusters above or below the threshold. Default is 0 ie
       clusters above.
    mask
    connectivity
    two_sample
    min_cluster_size

    Returns
    ---------------------
    cluster_image:    a numpy.nd array,
              with the same size as the test-statistic in which the clusters
              above the CDT are labelled each with a different number.

    Examples
    ---------------------
    # Clusters above 0.5
    cluster_image, cluster_sizes = pr.find_clusters(np.array([[1,0,1],[1,1,0]]), 0.5)
    # Clusters below 0.5
    cluster_image, cluster_sizes = pr.find_clusters(np.array([[1,0,1],[1,1,0]]), 0.5, below = 1)
    # tstat image
    f = pr.statnoise((50,50), 20, 10)
    tstat, xbar, std_dev = pr.mvtstat(f.field)
    cluster_image, c_sizes = pr.find_clusters(tstat, 2)
    plt.imshow(cluster_image)
    """

    # Mask the data if that is possible
    if np.sum(np.ravel(mask)) > 0:
        test_statistic = test_statistic*mask

    if two_sample:
        raise Exception("two sample hasn't been implemented yet!")

    if below:
        cluster_image = measure.label((test_statistic < cdt)*(test_statistic > 0),
                                      connectivity=connectivity)
    else:
        cluster_image = measure.label(
            test_statistic > cdt, connectivity=connectivity)

    n_clusters = np.max(cluster_image)
    store_cluster_sizes = np.zeros(1)

    # Use J to keep track of the clusters
    j = 0

    for i in np.arange(n_clusters):
        cluster_index = (cluster_image == (i+1))
        cluster_size = np.sum(cluster_index)
        if cluster_size < min_cluster_size:
            cluster_image[cluster_index] = 0
        else:
            j = j + 1
            store_cluster_sizes = np.append(store_cluster_sizes, cluster_size)
            cluster_image[cluster_index] = j

    # Remove the initial zero
    store_cluster_sizes = store_cluster_sizes[1:]

    return cluster_image, store_cluster_sizes


def cluster_tdp(data, design, contrast_matrix, mask, n_bootstraps=100, alpha=0.1,
                min_cluster_size=30, cdt=0.001, method='boot'):
    """ cluster_tdp calculates the TDP (true discovery proportion) within
    clusters of the test-statistic.
  Parameters
  ---------------------
  imgs
  design
  contrast_matrix

  Returns
  ---------------------
  cluster_image:    a numpy.nd array,
              with the same size as the test-statistic in which the clusters
              above the CDT are labelled each with a different number.

  Examples
  ---------------------
    """
    # Obtain the number of parameters in the model
    n_params = contrast_matrix.shape[1]

    # Obtain the number of contrasts
    n_contrasts = contrast_matrix.shape[0]

    # Convert the data to a field
    data = pr.make_field(data)

    # Obtain the number of subjects
    nsubj = data.fibersize

    # Obtain the test statistics and convert to p-values
    test_stats, _ = pr.contrast_tstats(data, design, contrast_matrix)
    pvalues = 2*(1 - t.cdf(abs(test_stats.field), nsubj-n_params))

    # Perform Post-hoc inference
    if method == 'boot':
        # Run the bootstrapped algorithm
        _, _, pivotal_stats, _ = pr.boot_contrasts(data, design,
                                                   contrast_matrix, n_bootstraps=n_bootstraps, display_progress=1)

        # Obtain the lambda calibration
        lambda_quant = np.quantile(pivotal_stats, alpha)
    else:
        lambda_quant = alpha

    # Calculate the number of voxels in the mask
    n_vox_in_mask = np.sum(data.mask[:])

    # Gives t_k^L(lambda) = lambda*k/m for k = 1, ..., m
    thr = sa.linear_template(lambda_quant, n_vox_in_mask, n_vox_in_mask)

    # Calculate the TDP within each cluster

    # Initialize the matrix to store the tdp
    tdp_bounds = np.zeros(pvalues.shape)

    # Convert the mask to logical
    mask = mask > 0

    # For each cluster calculate the TDP
    for l in np.arange(n_contrasts):
        # Get the clusters of the test-statistic
        cluster_im, cluster_sizes = pr.find_clusters(pvalues[..., l], cdt, below=1,
                                                     mask=mask, min_cluster_size=min_cluster_size)

        # Obtain the number of clusters
        n_clusters = len(cluster_sizes)

        for i in np.arange(n_clusters):
            # Obtain the logical entries for where each region is
            region_idx = cluster_im == (i+1)

            # Compute the TP bound
            bound = sa.max_fp(pvalues[region_idx, l], thr)
            print(region_idx.shape)
            print(tdp_bounds[region_idx, l].shape)
            tdp_bounds[region_idx, l] = (
                np.sum(region_idx) - bound) / np.sum(region_idx)

    return tdp_bounds


def cluster_tdp_brain(imgs, design, contrast_matrix, mask, n_bootstraps=100, fwhm=4,
                      alpha=0.1, min_cluster_size=30, cdt=0.001, simtype=1, template='linear', do_sd=1):
    """ cluster_tdp_brain calculates the TDP (true discovery proportion) within
    clusters of the test-statistic. This is specifically for brain images
    and enables plotting of these images using the nilearn toolbox
    Parameters
    ---------------------
    imgs
    design
    contrast_matrix
    savedir

    Returns
    ---------------------
    cluster_image:    a numpy.nd array,
              with the same size as the test-statistic in which the clusters
              above the CDT are labelled each with a different number.

    Examples
    ---------------------
    """
    # Obtain the number of parameters in the model
    n_params = contrast_matrix.shape[1]

    # Obtain the number of contrasts
    n_contrasts = contrast_matrix.shape[0]

    # Obtain the template function
    t_func, _, _ = pr.t_ref(template)

    # Load the data
    masker = NiftiMasker(smoothing_fwhm=fwhm, mask_img=mask,
                         memory='/storage/store2/work/sdavenpo/').fit()
    data = masker.transform(imgs).transpose()

    # Convert the data to a field
    data = pr.make_field(data)

    # Obtain the number of subjects
    nsubj = data.fibersize

    if not len(imgs) == nsubj:
        raise Exception(
            "The number of subjects in imgs doesn't match the number within the data")

    # Obtain the test statistics and convert to p-values
    test_stats, _ = pr.contrast_tstats(data, design, contrast_matrix)
    #pvalues = 2*(1 - t.cdf(abs(test_stats.field), nsubj-n_params))
    pvalues = pr.tstat2pval(test_stats.field, nsubj-n_params)

    # Load the mask
    mask = load_img(mask).get_fdata()

    # Obtain a 3D brain image of the p-values for obtaining clusters
    # (squeezing to remove the trailing dimension)
    pvalues_3d = np.squeeze(
        get_data(masker.inverse_transform(pvalues.transpose())))

    # Calculate the number of voxels in the mask
    n_vox_in_mask = np.sum(mask[:])

    # Calculate the total number of null hypotheses
    m = n_contrasts*n_vox_in_mask

    # Perform Post-hoc inference
    if simtype == 1:
        # Run the bootstrapped algorithm
        _, _, pivotal_stats, bootstore = pr.boot_contrasts(data, design, contrast_matrix,
                                                           n_bootstraps=n_bootstraps, display_progress=1, store_boots=do_sd)

        # Obtain the lambda calibration
        lambda_quant = np.quantile(pivotal_stats, alpha)

        if do_sd:
            lambda_quant_sd, _ = pr.step_down(bootstore, alpha=alpha,
                                              do_fwer=0, template=template)
    else:
        # Calculate the lambda quantile for JER control
        lambda_quant = alpha
        if do_sd:
            hommel_value = pr.compute_hommel_value(np.ravel(pvalues), alpha)

            # Calculate the lambda quantile for JER control (with stepdown)
            lambda_quant_sd = lambda_quant/(hommel_value/m)

    # Gives t_k(lambda) = lambda*k/m for k = 1, ..., m
    thr = t_func(lambda_quant, m, m)

    if do_sd == 1:
        thr_sd = t_func(lambda_quant_sd, m, m)

    # Calculate the TDP within each cluster
    # Initialize the matrix to store the tdp
    tdp_bounds = np.zeros(pvalues_3d.shape)
    if do_sd:
        tdp_bounds_sd = np.zeros(pvalues_3d.shape)
    else:
        tdp_bounds_sd = -1

    # Convert the mask to logical
    mask = mask > 0

    # For each cluster calculate the TDP
    for l in np.arange(n_contrasts):
        # Get the clusters of the test-statistic
        cluster_im, cluster_sizes = pr.find_clusters(pvalues_3d[..., l], cdt,
                                                     below=1, mask=mask, min_cluster_size=min_cluster_size)

        # Obtain the number of clusters
        n_clusters = len(cluster_sizes)

        for i in np.arange(n_clusters):
            # Obtain the logical entries for where each region is
            region_idx = cluster_im == (i+1)

            # Compute the TP bound
            bound = sa.max_fp(pvalues_3d[region_idx, l], thr)
            print(region_idx.shape)
            print(tdp_bounds[region_idx, l].shape)
            tdp_bounds[region_idx, l] = (
                np.sum(region_idx) - bound)/np.sum(region_idx)

            # Compute the step down TP bound
            if do_sd:
                bound_sd = sa.max_fp(pvalues_3d[region_idx, l], thr_sd)
                tdp_bounds_sd[region_idx, l] = (
                    np.sum(region_idx) - bound_sd)/np.sum(region_idx)

    return tdp_bounds, tdp_bounds_sd, lambda_quant, lambda_quant_sd, masker
