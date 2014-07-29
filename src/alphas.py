# Python, np, scipy modules
# from plotting import AlphasRunningPlot
import sys
import numpy as np
from ext.configobj import ConfigObj

# Alphas fitter modules
from src.dataset import GobalDataSet
from src.providers import DataProvider
from src.fitter import MinuitFitter
from src.plotting import DataTheoryRatioPlot


def calculate_chi2(**kwargs):

    global_config = ConfigObj(kwargs)
    if kwargs['config']:
        config_file = ConfigObj(kwargs['config'])
        global_config.update(dict((k, v) for k, v in config_file.items() if global_config[k] is None))

    # Global dataset holding all data points, covariance matrices, etc...
    global_dataset = GobalDataSet()
    for dataset_filename in global_config['datasets']:
        dataset_provider = DataProvider(dataset_filename, global_config)
        dataset = dataset_provider.get_dataset()
        dataset.set_theory_parameters(kwargs['asmz'])
        global_dataset.add_dataset(dataset)

    # We calculate the Chi2 using a 'fit', but we fix alphasmz
    # Fit is needed if some nuisance parameters need to be fitted.
    # chi2_calculator = Chi2Nuisance(global_dataset)
    # print chi2_calculator.get_chi2()
    # print chi2_calculator.get_nuisance_parameters()

    #We don't want to fit asmz
    props = {'asmz': kwargs['asmz'], 'fix_asmz': True}
    fitter = MinuitFitter(global_dataset, user_initial_values=props)
    fitter.do_fit()


def perform_fit(**kwargs):
    """ Performs a alphas fit with the supplied datasets
    Kwargs:
    """
    global_config = ConfigObj(kwargs)
    if kwargs['config']:
        config_file = ConfigObj(kwargs['config'])
        global_config.update(dict((k, v) for k, v in config_file.items() if global_config[k] is None))

    # Global dataset holding all data points, covariance matrices, etc...
    global_dataset = GobalDataSet()
    for dataset_filename in global_config['datasets']:
        dataset_provider = DataProvider(dataset_filename, global_config)
        dataset = dataset_provider.get_dataset()
        dataset.set_theory_parameters(asmz=0.118)
        global_dataset.add_dataset(dataset)

    fitter = MinuitFitter(global_dataset)
    fitter.do_fit()
    # fit.save_result()


def plot_d2t(**kwargs):
    """Produce the interesting plots, dependent on set commandline options"""
    global_config = ConfigObj(kwargs)
    if kwargs['config']:
        config_file = ConfigObj(kwargs['config'])
        global_config.update(dict((k, v) for k, v in config_file.items() if global_config[k] is None))

    for dataset_filename in global_config['datasets']:
        dataset_provider = DataProvider(dataset_filename, global_config)
        dataset = dataset_provider.get_dataset()
        dataset.set_theory_parameters(kwargs['asmz'])
        d2t_plot = DataTheoryRatioPlot(dataset)
        d2t_plot.do_plot()

