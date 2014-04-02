# Python, np, scipy modules
import os
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from plotting import AlphasRunningPlot
from ext.configobj import ConfigObj

# Alphas fitter modules
from measurement import GobalDataSet, TheoryCalculatorSource
from providers import DataSetProvider


def perform_fit(**kwargs):

    # read config
    analysis_config = ConfigObj(kwargs['config'])
    # read all datasets
    dataset_filenames = analysis_config['datasets'].as_list('dataset_filenames')

    global_dataset = GobalDataSet()

    for dataset_filename in dataset_filenames:
        dataset_provider = DataSetProvider(dataset_filename)
        dataset = dataset_provider.get_dataset()
        # Add user defined theory calculation source
        theory_source = TheoryCalculatorSource(label='asq_theory', origin='theory')
        dataset.add_sources([theory_source])
        global_dataset.add_dataset(dataset)

    fit = AlphasFitter(global_dataset)
    fit.do_fit()
    fit.save_result()
    print fit._asmz_fit
    print fit._asmz_fit_uncert


def plot(**kwargs):
    """Produce the interesting plots, dependent on set commandline options"""
    # read confi
    analysis_config = ConfigObj(kwargs['config'])
    # read all datasets
    datasets_filenames = analysis_config['datasets'].as_list('dataset_filenames')

    datasets = []
    for dataset_filename in datasets_filenames:
        dataset_provider = DataSetProvider(dataset_filename)
        dataset = dataset_provider.get_dataset()
        datasets.append(dataset)

    as_plot = AlphasRunningPlot(datasets)
    as_plot.do_plot()


# def get_chi2(data, theory, cov_matrix):
#     """Simple definition to calculate chi2 using covariance matrix."""
#     inv_matrix = np.matrix(cov_matrix).getI()
#     residual = np.matrix(data - theory)
#     chi2 = (residual * inv_matrix * residual.getT())[0, 0]
#     return chi2


class AlphasFitter(object):

    def __init__(self, dataset):

        self._asmz_fit = None
        self._asmz_fit_uncert = None
        self._tolerance = 1.0
        self._bounds = (0.100, 0.200)

        self._dataset = dataset


    @staticmethod
    def _min_func(asmz, dataset):
        """Function to be minimized.
           Here a basically the chi2 of the datasets is minimized by taking into account all correlations
           between different datasets and within the datasets themselves.
        """
        dataset.set_theory_parameters(asmz=asmz)
        return dataset.get_chi2()

    @staticmethod
    def _root_func(asmz, dataset, asmz_central, tolerance=1.0):
        """ Function used to evaluate the parameter uncertainty. min_func(x) is shifted downwards using
            the f(best_fit) and the tolerance. The roots of this function are evaluated and define the
            parameter uncertainty using the given tolerance.
        """
        chi2 = (AlphasFitter._min_func(asmz, dataset) -
                AlphasFitter._min_func(asmz_central, dataset) -
                tolerance)
        return chi2

    def do_fit(self):
        # Chi2 tolerance for error evaluation
        fit_result = minimize_scalar(AlphasFitter._min_func,
                                     args=(self._dataset,),
                                     method='bounded',
                                     bounds=self._bounds)
        self._asmz_fit = fit_result.x
        self._calc_par_uncert()
        # Find root of function min_func - asmz_central + tolerance

    def _calc_par_uncert(self):
        asmz_l = brentq(AlphasFitter._root_func, self._bounds[0], self._asmz_fit,
                        args=(self._dataset, self._asmz_fit, self._tolerance))
        asmz_h = brentq(AlphasFitter._root_func, self._asmz_fit, self._bounds[1],
                        args=(self._dataset, self._asmz_fit, self._tolerance))

        self._asmz_fit_uncert = (self._asmz_fit - asmz_l, asmz_h - self._asmz_fit)

    def save_result(self, filepath=None):
        if filepath is None:
            filepath = os.path.join('output/', 'Result.txt')
        with open(filepath, 'w') as f:
            f.write('# {} {} {} {}\n'.format('q', 'asq', 'tot_l', 'tot_h'))
            f.write('{} {} {} {}'.format(91.18, self._asmz_fit, *self._asmz_fit_uncert))
