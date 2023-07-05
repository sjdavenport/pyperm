"""
Additional functions for the sanssouci toolbox
"""
import sanssouci as sa
from scipy.stats import beta
import numpy as np
import pyperm as pr


def t_beta(lamb, k, n_hypotheses):
    """ A function to compute the template for the beta family

    Parameters
    -----------------
    lamb   double,
          the lambda value to evaluate at
    k: int,
          indexes the reference family
    m: int,
          the total number of p-values

    Returns
    -----------------
    a numpy.ndarray of shape ()

    Examples
    -----------------
    lamb = 0.1
    pr.t_beta(lamb, 1, 5)

    # Plot the beta curves
    import matplotlib.pyplot as plt
    i = 1000;
    lamb_vec = np.arange(i)/i
    plt.plot(lamb_vec, pr.t_beta(lamb_vec, 1, 10))

    lamb = 0.9; m = 1000; k = np.arange(m)
    plt.plot(k, pr.t_beta(lamb, k, m))

    lamb = 0.001; m = 1000; k = np.arange(100)
    plt.plot(k, pr.t_beta(lamb, k, m))

    lamb = np.exp(-10); m = 50; k = np.arange(m)
    plt.plot(k, pr.t_beta(lamb, k, m))
    """
    # t_k^B(lambda) = F^{-1}(lambda) where F (beta.ppf) is the cdf of the
    # beta(k, m+1-k) distribution. This yields the lambda quantile of this distribution
    return beta.ppf(lamb, k, n_hypotheses+1-k)


def t_inv_beta(set_of_pvalues):
    """ A function to compute the inverse template for the beta family

    Parameters
    -----------------
    p0: a numpy.ndarray of shape (B,m) ,
          where m is the number of null hypotheses and B is typically the number
          of permutations/bootstraps that contains the values on which to apply (column wise)
          the inverse beta reference family

    Returns
    -----------------
    a numpy.ndarray of shape ()

    Examples
    data = np.random.uniform(0,1,(10,10))
    pr.t_inv_beta(data)
    -----------------
    """
    # Obtain the number of null hypotheses
    n_hypotheses = set_of_pvalues.shape[1]

    # Initialize the matrix of transformed p-values
    transformed_pvalues = np.zeros(set_of_pvalues.shape)

    # Transformed each column via the beta pdf
    # (t_k^B)^{-1}(lambda) = F(lambda) where F (beta.pdf) is the cdf of the
    # beta(k, m+1-k) distribution.
    for k in np.arange(n_hypotheses):
        transformed_pvalues[:, k] = beta.cdf(
            set_of_pvalues[:, k], k+1, n_hypotheses+1-(k+1))

    return transformed_pvalues


def t_ref(template='linear'):
    """ A function to compute the inverse template for the beta family

    Parameters
    -----------------
    template: str,
          a string specifying the template to use, the options are 'linear' (default)
          and 'beta'

    Returns
    -----------------
    t_func:  function,

    t_inv: function,


    Examples
    -----------------
    % Obtaining the linear template functions
    t_func, t_inv = t_ref()

    % Obtaining the beta template functions
    t_func, t_inv = t_ref('beta')
    """
    if template == 'linear' or template == 'simes':
        t_func = sa.linear_template
        t_inv = sa.inverse_linear_template
        t_inv_all = pr.inverse_linear_template_all
    elif template == 'beta' or template == 'b':
        t_func = sa.beta_template
        t_inv = sa.inverse_beta_template
    else:
        raise Exception(
            'The specified template is not available or has been incorrectly input')

    return t_func, t_inv, t_inv_all


def inverse_linear_template_all(pvals, K, do_sort=False):
    """
    A function to compute t_k^(-1)(pvals) for k = 1, \dots, K, where t_k(x) = xk/K.
    Note that the pvals typically need to be sorted before input to this function!
    ----------------------------------------------------------------------------
    ARGUMENTS
    - pvals: np.ndarry,
        an array of size B by m (B: nperm, m: number of null hypotheses)
    - K:  int,
        an integer giving the size of the reference family
    ----------------------------------------------------------------------------
    OUTPUT
    - out:   a numpy array such that out_{bn} = p0_{bn}*K/n 
    ----------------------------------------------------------------------------
    EXAMPLES
      from scipy.stats import norm
      pvals = norm.cdf(np.random.randn(5))
      pvals = np.sort(pvals)
      out =  pr.inverse_linear_template(pvals, 5)
    ----------------------------------------------------------------------------
    """
    # Sort the pvalues unless otherwise specified
    if do_sort:
        pvals = np.sort(pvals, axis=1)

    # Obtain the number of columns of pvals: the total number of hypotheses being tested
    if len(pvals.shape) == 1:
        m = len(pvals)
    elif len(pvals.shape) == 2:
        m = pvals.shape[1]

    # Generate a vector: (1/m,2/m,...,1)
    normalized_ranks = (np.arange(m)+1)/float(K)

    return pvals/normalized_ranks


def get_pivotal_stats(pval_matrix, size_of_original_template, template='linear'):
    """A function to obtain the pivotal statistics given observed p-values

    Parameters
    ----------
    pval_matrix:  a numpy.nd array,
        of size (B, m) where B is the number of permutations and m is the number
        of hypotheses.
    size_of_original_template:  int
        the size of the original template (note that the pvalue matrix may be a
        subset of the original data, i.e. when running a step down algorithm 
        so this value may not equal that of the size of the data)
    template: char,
        a character array giving the template type. Default is 'linear'.

    Returns
    -------
    array-like of shape (B,)
        A numpy array of of size [B]  containing the pivotal statistics, whose
        j-th entry corresponds to \psi(g_j.X) with notation of the Blanchard et al 2020.

    Examples
    ----------
    # Comparing to the implementation in the sanssouci package
    from scipy.stats import norm
    pvals = norm.cdf(np.random.randn(5,10))
    out1 =  pr.get_pivotal_stats(pvals, 10)
    print(out1)
    out2 = sa.get_pivotal_stats(pvals, sa.inverse_linear_template)
    print(out2)

    References
    ----------
    [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.
    """
    if isinstance(template, str):
        _, _, t_inv_all = pr.t_ref(template)
    else:
        raise Exception('The template must be input as a string')

    # Sort permuted p-values (within rows)
    pval_matrix = np.sort(pval_matrix, axis=1)

    # Apply the template function
    # For each feature p, compare sorted permuted p-values to template
    template_inverse_of_pvals = t_inv_all(
        pval_matrix, size_of_original_template)

    # Compute the minimum within each row
    pivotal_stats = np.min(template_inverse_of_pvals, axis=1)

    return pivotal_stats


def capstr(s):
    """Capitalizes the first letter of each word in the input string.

    Parameters:
    s (str): The input string.

    Returns:
    str: The input string with the first letter of each word capitalized.

    Examples:
    capitalize_words("this is a test string")
    """

    # Split the string into a list of words
    words = s.split()

    # Capitalize the first letter of each word
    capitalized_words = [word[0].upper() + word[1:].lower() for word in words]

    # Join the list of capitalized words back into a single string
    return ' '.join(capitalized_words)
