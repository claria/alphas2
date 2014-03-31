# Python, np, scipy modules
import os
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from plotting import AlphasRunningPlot, ProfileLikelihoodPlot

# Alphas fitter modules
from measurement import MetaDataSet, TheoryCalculatorSource
from providers import DataSetProvider
from config import config


def perform_fit(**kwargs):

    metadataset = prepare_dataset(**kwargs)
    result = minimize_scalar(min_func, args=(metadataset,),
                             method='bounded',
                             bounds=(0.1000, 0.2000))

    # Chi2 tolerance for error evaluation
    tolerance = 1.0

    asmz_c = result.x
    # Find root of function min_func - asmz_central + tolerance
    asmz_l = asmz_c - brentq(rootfunc, 0.110, asmz_c, args=(metadataset,asmz_c, tolerance))
    asmz_h = brentq(rootfunc, asmz_c, 0.200, args=(metadataset, asmz_c, tolerance)) - asmz_c

    asmz = [91.18, asmz_c, asmz_l, asmz_h]
    save_result(asmz)


def rootfunc(asmz, metadataset, asmz_central, tolerance):
    print metadataset.datasets
    chi2 = min_func(asmz, metadataset) - min_func(asmz_central, metadataset) - tolerance
    print "xzy", chi2
    return chi2


def prepare_dataset(**kwargs):

    # read confi
    analysis_config = config.get_config(kwargs['config'])
    # read all datasets
    datasets = analysis_config['datasets'].as_list('dataset_filenames')
    print datasets

    metadataset = MetaDataSet()

    for dataset_filename in datasets:
        dataset_provider = DataSetProvider(dataset_filename)
        dataset = dataset_provider.get_dataset()
        # Add user defined theory calculation source
        theory_source = TheoryCalculatorSource(label='asq_theory', origin='theory')
        dataset.add_sources([theory_source])
        metadataset.add_dataset(dataset)

    return metadataset


def save_result(asmz):
    output_path = os.path.join(config.output_dir, 'result_asmz.txt')
    with open(output_path, 'w') as f:
        f.write('# {} {} {} {}\n'.format('q', 'asq', 'tot_l', 'tot_h'))
        f.write('{} {} {} {}'.format(*asmz))


def plot(**kwargs):
    """Produce the interesting plots, dependent on set commandline options"""
    # read confi
    analysis_config = config.get_config(kwargs['config'])
    # read all datasets
    datasets_filenames = analysis_config['datasets'].as_list('dataset_filenames')

    datasets = []
    for dataset_filename in datasets_filenames:
        dataset_provider = DataSetProvider(dataset_filename)
        dataset = dataset_provider.get_dataset()
        datasets.append(dataset)

    as_plot = AlphasRunningPlot(datasets)
    as_plot.do_plot()


def min_func(asmz, metadataset):
    """Function to be minimized"""
    for dataset in metadataset.datasets:
        dataset.get_source(label='theory').set_asmz(asmz)
    chi2 = get_chi2(metadataset.get_data(),
                    metadataset.get_theory(),
                    metadataset.get_cov_matrix())

    # print metadataset.get_nbins()
    return chi2


def get_chi2(data, theory, cov_matrix):
    """Simple definition to calculate chi2 using covariance matrix."""
    inv_matrix = np.matrix(cov_matrix).getI()
    residual = np.matrix(data - theory)
    chi2 = (residual * inv_matrix * residual.getT())[0, 0]
    return chi2



