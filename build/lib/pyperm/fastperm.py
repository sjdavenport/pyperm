import numpy as np
import pyperm as pr


def block_tstat(block_sum, block_sos, nsubj):
    """
    Computes one-sample t-statistic, mean, standard deviation and Cohen's d
    from the mean and sum of squares of different blocks/partitions.

    Parameters:
    block_sum (np.ndarray): An array of dimension dim x nblocks, where dim is
        the vector dimension, and each block_sum[:, ..., :, I] is the mean of
        the ith block.
    block_sos (np.ndarray): An array of dimension dim x nblocks, where dim is
        the vector dimension, and each block_sos[:, ..., :, I] is the sum of
        squares of the ith block.
    nsubj (int): The number of original subjects.

    Returns:
    ----------
    tuple: A tuple containing the following elements in order:
        tstat (np.ndarray): The one-sample t-statistic at each voxel.
        xbar (np.ndarray): The mean at each voxel.
        std_dev (np.ndarray): The standard deviation at each voxel, calculated
                            using the unbiased estimate of the variance.
        cohensd (np.ndarray): The Cohen's d at each voxel.

    Examples
    ----------
    nvox = 10
    nsubj = 100
    data = np.random.randn(nvox, nsubj)
    tstat_orig,_,_ = pr.mvtstat(data)
    block_sum, block_sos = block_summary_stats(data, 10)
    tstat_block,_,_,_ = block_tstat( block_sum, block_sos, nsubj )
    """
    s_block_sum = block_sum.shape
    D = len(s_block_sum) - 1

    # Compute the mean
    xbar = (1/nsubj)*np.sum(block_sum, axis=D)

    # Compute the (biased) estimate of the variance
    biased_variance = (1/nsubj)*np.sum(block_sos, axis=D) - xbar**2

    # Compute the standard deviation using the unbiased estimator of the
    # variance
    std_dev = np.sqrt((nsubj/(nsubj-1))*biased_variance)

    # Compute Cohen's d
    cohensd = xbar/std_dev

    # Compute the t-statistic
    tstat = cohensd*np.sqrt(nsubj)

    return tstat, xbar, std_dev, cohensd


def block_summary_stats(data, nblocks):
    """
    Computes the one-sample block summary statistics of the data for use in the
                            implementation of fast permutation techniques.

    Parameters
    ----------
    data: array_like
        a matrix of size [dim, nsubj]
    nblocks: int
        the number of blocks to divide the data into

    Returns
    -------
    block_sum: ndarray
        an array of size [dim, nblocks] such that block_sum(..., I) is the
                                            voxelwise sum within the Ith block
    block_sos: ndarray
        an array of size [dim, nblocks] such that block_sos(..., I) is the
                voxelwise sum of the squares of the entries of the Ith block

    Examples
    -------
    nvox = 10
    nsubj = 100
    data = np.random.randn(nvox, nsubj)
    block_sum, block_sos = block_summary_stats(data, 10)
    """
    s_data = data.shape
    nsubj = s_data[-1]
    dim = s_data[:-1]
    D = len(dim)

    nsubj_per_block = nsubj//nblocks
    if nsubj_per_block < 1:
        print('There is less than one subject per block so defaulting to a block size of 1 subject')
        nsubj_per_block = 1
        nblocks = nsubj

    block_sum = np.zeros([*dim, nblocks])
    block_sos = np.zeros([*dim, nblocks])

    block_subject_indices = np.arange(0, nsubj_per_block) - nsubj_per_block
    for i in range(nblocks):
        block_subject_indices = block_subject_indices + nsubj_per_block

        if i == nblocks-1:
            block_subject_indices = block_subject_indices[block_subject_indices <= nsubj]

        block_sum[:, i] = np.sum(data[:, block_subject_indices], axis=-1)

        block_sos[:, i] = np.sum(
            np.power(data[:, block_subject_indices], 2), axis=-1)

    return block_sum, block_sos


def block_lm_summary_stats(data, design, nblocks, doblockY=False):
    """
    Compute summary statistics for a time series data block by block.

    Parameters:
    data (list or numpy.ndarray): a 1D time series data.
    block_size (int, optional): the size of each block. Default is 100.

    Returns:
    list: a list of dictionaries, where each dictionary contains the following
        keys: 'block_start', 'block_end', 'mean', 'median', 'min', 'max', 
        'range', 'standard deviation'. The values correspond to the start
        and end indices of each block, and the summary statistics for the
        data within each block.
    """
    # Check mandatory input and get important constants
    s_data = np.shape(data)
    nsubj = s_data[-1]
    dim = s_data[:-1]
    D = len(dim)

    # Calculate the number of parameters
    nparams = design.shape[1]

    # Compute the number of subjects per block
    nsubj_per_block = nsubj // nblocks
    if nsubj_per_block < 1:
        print("There is less than one subject per block so defaulting to a block size of 1 subject")
        nsubj_per_block = 1
        nblocks = nsubj

    # Initialize the arrays to store the block sum and sum of squares
    block_xY = np.zeros((*dim, nparams, nblocks))
    block_sos = np.zeros((*dim, nblocks))  # sos = sum of squares
    if doblockY:
        block_Y = np.zeros((*dim, nblocks))

    # Main Function Loop
    # Initialize the indices for the blocks
    block_subject_indices = np.arange(1, nsubj_per_block+1) - nsubj_per_block

    # Main loop
    for I in range(nblocks):
        block_subject_indices = block_subject_indices + nsubj_per_block

        # The last block may have less subjects if the blocks are not split evenly
        if I == nblocks-1:
            block_subject_indices = block_subject_indices[block_subject_indices <= nsubj]

        # Obtain the blocks of X^TY
        block_xY[..., :, I] = np.dot(
            data[..., block_subject_indices], design[block_subject_indices, :])

        # Assign the block sum of squares
        block_sos[..., I] = np.sum(
            np.power(data[..., block_subject_indices], 2), axis=-1)

    if doblockY:
        # Initialize the indices for the blocks
        block_subject_indices = np.arange(
            1, nsubj_per_block+1) - nsubj_per_block

        for J in range(nblocks):
            block_subject_indices = block_subject_indices + nsubj_per_block

            # Calculate the sum of the Ys
            block_Y[..., J] = np.sum(data[..., block_subject_indices], axis=-1)

        return block_xY, block_sos, block_Y
    else:
        return block_xY, block_sos
