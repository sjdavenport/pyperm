# -*- coding: utf-8 -*-
"""
A set of functions which facilitate plotting of the results
"""
import numpy as np
import sanssouci as sa
import matplotlib.pyplot as plt

def fdp_plot(pvalues, thr_list, labels, colors, linestyle, number2plot, saveloc, dolegend = 1, vertline = 0, vertlinestyle = '-', idlinestyle = 'dashed'):
    """ A function to compute the fdp and tdp confidence envelope plots

    Parameters
    -----------------
    pvalues:
        
    thr:
    
    number2plot:
        
    saveloc: str,
        a string which specifies a location in which to save the figures if 
        desired. Default is not to do so.

    Returns
    -----------------
    generates fdp and tdp plots

    Examples
    -----------------
    """
    npvals = len(pvalues)

    # Ensure that selected number is not greater than the total number of p-values
    number2plot = np.min([number2plot, npvals])
    
    # Sort the p-values and restrict to the lower ones
    pvalues_sorted = np.sort(np.ravel(pvalues))
    pvalue_subset = pvalues_sorted[:number2plot]
        
    # Generate the vector [0,...,npvals]
    one2npvals = np.arange(1, number2plot + 1)
    
    max_FDP = list()
    min_TP = list()
    maxminTP = 0
    for i in np.arange(len(thr_list)):
        # Calculate the confidence envelope for the number of false positives
        max_FP = sa.curve_max_fp(pvalue_subset, thr_list[i])

        # Dividing the envelope by the number of elements in the set gives a bound on the false discovery proportion
        max_FDP.append(max_FP[0: number2plot] / one2npvals[0: number2plot])
        min_TP.append(one2npvals[0: number2plot]  - max_FP[0: number2plot])
        
        maxminTP = np.max((maxminTP, np.max(min_TP[i])))

    # Initialize the figure
    figure = plt.figure(figsize=(10, 4))

    # Plot the false discovery proportion and its bound
    plt.subplot(121)
    if vertline > 0:
        plt.axvline(x = vertline, color = 'silver', linestyle = vertlinestyle)
    for i in np.arange(len(thr_list)):
        plt.plot(max_FDP[i], label=labels[i], color = colors[i], linestyle = linestyle[i], linewidth = 2)
    plt.title('Upper bound on FDP')
    plt.xlim(1, number2plot)
    plt.xlabel('k')
    plt.ylabel('FDP($p_{(1)}, \dots, p_{(k)}$)')
    #plt.legend(loc="lower right")

    # Plot the true postives and their bound
    plt.subplot(122)
    plt.plot(np.arange(np.ceil(1.1*maxminTP)), np.arange(np.ceil(1.1*maxminTP)), label = 'identity', linestyle = idlinestyle, color = 'black')
    if vertline > 0:
        plt.axvline(x = vertline, color = 'silver', linestyle = vertlinestyle)
    for i in np.arange(len(thr_list)):
        plt.plot(min_TP[i], label=labels[i], color = colors[i], linestyle = linestyle[i], linewidth = 2)
    plt.title('Lower bound on TP')
    if dolegend == 1:
        plt.legend(loc="lower right", fontsize = '12')
    plt.xlim(1, number2plot)
    plt.ylim(0, np.ceil(1.1*maxminTP))
    plt.xlabel('k')
    plt.ylabel('TP($p_{(1)}, \dots, p_{(k)}$)')
    #figure, axes = plt.subplots(nrows=1, ncols=2) 
    figure.tight_layout(pad=1.0)
    
    saveloc += '_' + str(number2plot) + '.pdf'
    plt.savefig(saveloc)
