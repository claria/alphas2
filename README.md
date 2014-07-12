# alphas2
=======

## Overview


Tool to fit alpha_S from cross section measurements using fastNLO. Data with all
uncertainties including all correlation is provided using plain text files and are
steered by configuration files.




## Installation:
=============
------------
Dependencies:

Python >= 2.6

Iminuit

fastnlo_toolkit

You need to install the fastNLO toolkit with python bindings and setup the LD_LIBRARY_PATH
 and PYTHONPATH accordingly so that python finds the fastnlo module.

FNLO="/home/aem/uni/sw/fnlo_toolkit/"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FNLO/lib
export PYTHONPATH=$PYTHONPATH:$FNLO/lib/python2.7/site-packages

Additionally you need to install iminuit, a convenient and pythonic interface to
the SEAL MINUIT package. Iminuit can either be found in the package repository of
your system or on Pypi.

https://pypi.python.org/pypi/iminuit/1.1.1

## Usage:
======

