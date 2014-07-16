# alphas2.py
=========

## Overview
===========

Tool to fit alpha_S from cross section measurements using fastNLO. Data with all
uncertainties including all correlation is provided using plain text files and are
steered by configuration files.

## Installation:
================
Dependencies:

Python >= 2.6

Iminuit >= 1.1.1

fasNLO Toolkit >= 2.1

You need to install the fastNLO toolkit with python bindings and setup the LD_LIBRARY_PATH
and PYTHONPATH accordingly so that python finds the fastnlo module.

http://fastnlo.hepforge.org/

FNLO="/home/aem/uni/sw/fnlo_toolkit/"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FNLO/lib
export PYTHONPATH=$PYTHONPATH:$FNLO/lib/python2.7/site-packages

Additionally you need to install iminuit, a convenient and pythonic interface to
the SEAL MINUIT package. Iminuit can either be found in the package repository of
your system or on Pypi. MINUIT is already included in the iminuit package.

https://pypi.python.org/pypi/iminuit/1.1.1

## Usage:
=========

### Description of the datasets
===============================

### Chi2 calculation:
=====================

General comparisons between data and theory can be done by calculating the Chi2. alphas2.py can calculate the
chi2 while taking into account all kind of correlations. You have to


To calculate the chi2 for a dataset you can use the following command. fastNLO uses the PDF set PDF_SET.LHgrid
and the value of the strong coupling at Mz of ASMZ to calculate the NLO prediction. The measurement and all uncertainties
specified in the DATASET.txt will be considered when calculating the chi2:

```
alphas2/run.py -p PDF_Set.LHgrid -d data/DATASET.txt -a ASMZ
```


