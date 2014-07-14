# Python, np, scipy modules
# from plotting import AlphasRunningPlot
import sys
from ext.configobj import ConfigObj

# Alphas fitter modules
from src.dataset import GobalDataSet
from src.providers import DataProvider
from src.fitter import MinuitFitter


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

    print global_dataset.get_chi2()


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

    print global_dataset.get_data()/global_dataset.get_theory()
    fitter = MinuitFitter(global_dataset)
    fitter.do_fit()
    # fit.save_result()


def plot(**kwargs):
    """Produce the interesting plots, dependent on set commandline options"""
    # read confi
    # analysis_config = ConfigObj(kwargs['config'])
    # read all datasets
    # datasets_filenames = analysis_config['datasets'].as_list('dataset_filenames')
    #
    # datasets = []
    # for dataset_filename in datasets_filenames:
    #     dataset_provider = DataSetProvider(dataset_filename)
    #     dataset = dataset_provider.get_dataset()
    #     datasets.append(dataset)
    #
    # as_plot = AlphasRunningPlot(datasets)
    # as_plot.do_plot()
    pass
