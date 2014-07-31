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
        dataset_provider = DataProvider(dataset_filename, global_config['pdfset'])
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
        dataset_provider = DataProvider(dataset_filename, global_config['pdfset'])
        dataset = dataset_provider.get_dataset()
        dataset.set_theory_parameters(asmz=0.118)
        global_dataset.add_dataset(dataset)

    fitter = MinuitFitter(global_dataset)
    fitter.do_fit()
    # fit.save_result():w


def plot_d2t(**kwargs):
    """Produce the interesting plots, dependent on set commandline options"""
    global_config = ConfigObj(kwargs)
    if kwargs['config']:
        config_file = ConfigObj(kwargs['config'])
        global_config.update(dict((k, v) for k, v in config_file.items() if global_config[k] is None))

    for dataset_filename in global_config['datasets']:
        # Split plots at diff_0 bins
        pdfsets = global_config.as_list('pdfsets')
        _dataset_provider = DataProvider(dataset_filename, pdfsets[0])
        _dataset_config = _dataset_provider.get_dataset_config()
        _dataset = _dataset_provider.get_dataset()
        diff_0 = _dataset_config['plot']['diff_0']

        for diff_0_bin in _dataset.get_bin(diff_0).get_arr_mid_unique():
            datasets = []
            for pdfset in pdfsets:
                dataset_provider = DataProvider(dataset_filename, pdfset)
                dataset = dataset_provider.get_dataset()
                dataset.set_theory_parameters(kwargs['asmz'])
                dataset.apply_cut('{}_eq'.format(diff_0), diff_0_bin)
                datasets.append(dataset)
            d2t_plot = DataTheoryRatioPlot(datasets, output_fn='output/plots/d2t/{0}/{0}_{1}_{2}'.format(_dataset.label,
                                                                                                         diff_0,
                                                                                                         diff_0_bin))
            d2t_plot.do_plot()
