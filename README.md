# PyRFT (Beta version)
This package performs inference on in high-dimensional linear models using resampling method. In particular it provides post-hoc inference for multiple testing methods when considering multiple contrasts. This package is written together with the [SanSouci](https://github.com/pneuvial/sanssouci.python) python package which performs post-hoc inference when the exchangeability hypothesis holds. For further details on the methodology and relevant papers
see https://www.math.univ-toulouse.fr/~pneuvial/sanssouci.html and our upcoming pre-print.

## Table of contents
* [Getting Started](#folderstruct)
* [Code Structure](#codestruct)
    * [cluster_inference.py](#cinference)
    * [fdp.py](#fdp)
    * [permutation.py](#permutation)
    * [power.py](#power)
    * [random_field_generation](#rft)
* [Set Up](#setup)
    * [Dependencies](#dependencies)

## Getting Started <a name="folderstruct"></a>
The majority of the code is available in the pyrft subfolder. The best way to get started is to work through the Jupyter Notebook: [contrast_examples.ipynb](https://github.com/sjdavenport/pyrft/blob/master/examples/contrasts_example.ipynb) available in the Examples folder. 

## Code Structure <a name="codestruct"></a>
The code for this package is contained within the pyrft subfolder. This section contains a general description of the files with the most important functions.

### cluster_inference.py <a name="cinference"></a>
This file contains functions for calculating clusters and perform FDP inference on clusters.

### fdp.py <a name="fdp"></a>
This file contains functions for running the Benjamini-Hochberg procedure to control the FDR as well as functions to run step-down algorithms.

### permutation.py <a name="permutation"></a>
This file contains functions to run permutation and bootstrap resampling. 

### power.py <a name="power"></a>
This file contains functions to compare the power of bootstrap and parametric methods as well as to generate signal with random locations.

### random_field_generation.py <a name="rft"></a>
This file contains functions to generate noisy random fields.  

## Set Up <a name="setup"></a>
In order to install this package you'll need to download the package, go to the pyrft folder and run pip install. Many of the functions rely on code from the [SanSouci](https://github.com/pneuvial/sanssouci.python) python package so we recommend that you install that as well - this can be done similarly.  

If you have any difficulties getting this code to run or have any questions
feel free to get in touch with me at sdavenport(AT)ucsd.edu or via twitter [@BrainStatsSam](https://twitter.com/BrainStatsSam).
