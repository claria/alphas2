# Python, np, scipy modules
# from plotting import AlphasRunningPlot
from ext.configobj import ConfigObj

# Alphas fitter modules
from src.measurement import GobalDataSet
from src.providers import DataProvider
from src.fitter import MinuitFitter


def perform_fit(**kwargs):

    # read config
    global_config = ConfigObj(kwargs['config'])
    # read all datasets
    dataset_filenames = global_config.as_list('datasets')

    # Global dataset holding all data points, covariance matrices, etc...
    global_dataset = GobalDataSet()

    for dataset_filename in dataset_filenames:
        dataset_provider = DataProvider(dataset_filename, global_config)
        # dataset_config = dataset_provider.get_dataset_config()
        dataset = dataset_provider.get_dataset()

        global_dataset.add_dataset(dataset)

    fit = MinuitFitter(global_dataset)
    fit.do_fit()
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

# def get_chi2(data, theory, cov_matrix):
#     """Simple definition to calculate chi2 using covariance matrix."""
#     inv_matrix = np.matrix(cov_matrix).getI()
#     residual = np.matrix(data - theory)
#     chi2 = (residual * inv_matrix * residual.getT())[0, 0]
#     return chi2


