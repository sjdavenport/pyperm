"""
A file containing the random field generation functions
"""
import numpy as np
import pyperm as pr
from scipy.ndimage import gaussian_filter

def smooth(data, fwhm, mask = 0):
    """ smooth(data, fwhm, mask = 0) smoothes the components of the random 
    field data with an isotropic Gaussian kernel with given fwhm

    Parameters
    ---------------------
    data    an object of class field,
        giving the randomness 
    fwhm   an int,
        giving full width half maximum with which to smooth the data
    mask   a numpy.nd array,
        containing 0 and 1s with the same dimensions as the data which 
        specifices the mask

    Returns
    ---------------------
    An object of class field the components of which are the smooth random fields

    Examples
    ---------------------
    # 1D example
    f = pr.wfield(50,10)
    smooth_f = pr.smooth(f, 8)
    plt.plot(smooth_f.field)

    # 2D example
    f = pr.wfield((50,50), 10)
    smooth_f = pr.smooth(f, 8)
    plt.imshow(smooth_f.field[:,:,1])

    # 2D example with mask
    f = pr.wfield((50,50), 10)
    mask = np.zeros((50,50), dtype = 'bool')
    mask[15:35,15:35] = 1
    smooth_f = pr.smooth(f, 8, mask)
    plt.imshow(smooth_f.field[:,:,1])
    """
    # Convert a numpy array to a field if necessary
    if isinstance(data, np.ndarray):
        data = pr.make_field(data)

    # If a non-zero mask is supplied used this instead of the mask associated with data
    if np.sum(np.ravel(mask)) > 0:
        data.mask = mask

    # Calculate the standard deviation from the fwhm
    sigma = pr.fwhm2sigma(fwhm)

    for i in np.arange(data.fibersize):
        data.field[...,i] = gaussian_filter(data.field[...,i] * data.mask, sigma = sigma) * data.mask

    return data

def statnoise(mask, nsubj, fwhm, truncation = 1, scale_var = 1):
    """ statnoise constructs a an object of class Field with specified mask
    and fibersize and consisting of 2D or 3D stationary noise (arising from 
    white noise smoothed with a Gaussian kernel). 

    Parameters
    ---------------------
    mask:   a tuple or a Boolean array,
          If a tuple then it gives the size of the mask (in which case the mask
          is taken to be all true)
          If a Boolean array then it is the mask itself
          The mask must be 2D or 3D
    fibersize:   a tuple giving the fiber sizes (i.e. typically nsubj)

    Returns
    ---------------------
    An object of class field of stationary random noise

    Examples
    ---------------------
    # 2D
    Dim = (50,50); nsubj = 20; fwhm = 4
    F = pr.statnoise(Dim, nsubj, fwhm)
    plt.imshow(F.field[:,:,1])
    
    # 3D
    Dim = (50,50,50); nsubj = 20; fwhm = 4
    F = pr.statnoise(Dim, nsubj, fwhm)
    plt.imshow(F.field[:,:,25,1])
    
    # Plot the variance (the same everywhere up to noise because of the edge effect correction)
    plt.imshow(np.var(F.field, 2))
    np.var(F.field)

    # No smoothing example:
    Dim = (50,50); nsubj = 20; fwhm = 0
    F = pr.statnoise(Dim, nsubj, fwhm)
    plt.imshow(F.field[:,:,1])

    Notes
    ---------------------
    Need to adjust this to account for the edge effect!
    Also need to ensure that the field is variance 1!!
    """
    # Set the default dimension not to be 1D
    use1d = 0

    # If the mask is logical use that!
    if isinstance(mask, np.ndarray) and mask.dtype == np.bool:
        # If a logical array assign the mask shape
        masksize = mask.shape
    elif  isinstance(mask, tuple):
        # If a tuple generate a mask of all ones
        masksize = mask
        mask = np.ones(masksize, dtype = bool)
    elif isinstance(mask, int):
        use1d = 1
        masksize = (mask,1)
        mask = np.ones(masksize, dtype = bool)
    else:
        raise Exception("The mask is not of the right form")

    # Truncate to deal with edge-effects
    if truncation == 1:
        truncation = 4*np.ceil(fwhm)
        truncation = truncation.astype(int)
        
    # Calculate the overall size of the field
    if use1d:
        fieldsize = (masksize[0]+2*truncation,) + (nsubj,)
    else:
        t_masksize = np.asarray(masksize) + 2*truncation*np.ones(len(masksize))
        t_masksize = tuple(t_masksize.astype(int))
        fieldsize = t_masksize + (nsubj,)

    # Calculate the sigma value with which to smooth form the fwhm
    sigma = pr.fwhm2sigma(fwhm)

    # Generate normal random noise
    data = np.random.randn(*fieldsize)

    for n in np.arange(nsubj):
        data[...,n] = gaussian_filter(data[...,n], sigma = sigma)        
    
    if truncation > 0:
        if use1d:
            data = data[(truncation + 1):(masksize[0]+truncation+1), :]
        else:
             if len(masksize) == 2:
                 data = data[(truncation + 1):(masksize[0]+truncation+1), (truncation + 1):(masksize[1]+truncation+1), :]
             elif len(masksize) == 3:
                 data = data[(truncation + 1):(masksize[0]+truncation+1), (truncation + 1):(masksize[1]+truncation+1), (truncation + 1):(masksize[2]+truncation+1), :]
             else:
                 raise Exception("The mask must be 2D or 3D")
                 
    # Scale to ensure that the noise is variance 1!
    if scale_var and truncation > 0:
        if use1d:
            data = data/np.mean(np.std(data, 1, ddof = 1))
        else:
            #print(np.mean(np.std(data, 1, ddof = 1)))
            data = data/np.mean(np.std(data, len(masksize), ddof = 1))
    
    #print(np.mean(gaussian_filter(np.ones(fieldsize), sigma = sigma)))
    
    # Combine the data and the mask to make a field
    out = pr.Field(data, mask)

    # Return the output
    return out

def wfield(mask, fibersize, field_type = 'N', field_params = 3):
    """ wfield constructs a an object of class Field with specified mask
    and fibersize and consisting of white noise.

    Parameters
    ---------------------
    mask:   a tuple or a Boolean array,
          If a tuple then it gives the size of the mask (in which case the mask
          is taken to be all true)
          If a Boolean array then it is the mask itself
    fibersize:   a tuple giving the fiber sizes (i.e. typically nsubj)

    Returns
    ---------------------
    An object of class field with white noise

    Examples
    ---------------------
    example_field = pr.wfield(15, 10); print(example_field)
    example_field = pr.wfield((5,5), 10)


    Notes
    ---------------------
    Need to ensure that this function works in all settings, i.e. 1D masks specified
    as (10,1) for example! And under masks e.g.
    example_field = pr.wfield(np.array((0, 1, 1, 1, 0, 1, 1), dtype = 'bool'), 10)
    """

    # Set the default dimension not to be 1D
    use1d = 0

    # If the mask is logical use that!
    if isinstance(mask, np.ndarray) and mask.dtype == np.bool:
        # If a logical array assign the mask shape
        masksize = mask.shape
    elif  isinstance(mask, tuple):
        # If a tuple generate a mask of all ones
        masksize = mask
        mask = np.ones(masksize, dtype = bool)
    elif isinstance(mask, int):
        use1d = 1
        masksize = (mask,1)
        mask = np.ones(masksize, dtype = bool)
    else:
        raise Exception("The mask is not of the right form")

    # Calculate the overall size of the field
    if use1d:
        fieldsize = (masksize[0],) + (fibersize,)
    else:
        fieldsize = masksize + (fibersize,)

    # Generate the data from the specified distribution
    if field_type == 'N':
        data = np.random.randn(*fieldsize)

    # Combine the data and the mask to make a field
    out = pr.Field(data, mask)

    # Return the output
    return out
