# Python, np, scipy modules
import os
import numpy as np
from scipy.optimize import minimize_scalar
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
    print result
    asmz = [91.18, result.x, 0.0007, 0.0007]
    save_result(asmz)


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


def profile_likelihood(**kwargs):

    metadataset = prepare_dataset(**kwargs)

    asmz_range = np.arange(0.100, 0.140, 0.0001)
    chi2 = np.zeros(asmz_range.shape)

    for i, asmz in enumerate(asmz_range):
        chi2[i] = min_func(asmz=asmz, metadataset=metadataset)

    data = {'x' : asmz_range,
            'y' : chi2}

    profile_plot = ProfileLikelihoodPlot(data)
    profile_plot.do_plot()


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



