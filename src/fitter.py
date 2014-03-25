# Python, np, scipy modules
import numpy as np
from scipy.optimize import minimize_scalar
from plotting import AlphasRunningPlot

# Alphas fitter modules
from measurement import MetaDataSet
from providers import DataSetProvider
from measurement import Source
from config import config

# Python modules
from fastnlo import CRunDec


def perform_fit(**kwargs):

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

    result = minimize_scalar(min_func, args=(metadataset,),
                             method='bounded',
                             bounds=(0.1000, 0.2000))
    print result


def plot(**kwargs):
    """Produce the interesting plots, dependent on set commandline options"""
    # read confi
    analysis_config = config.get_config(kwargs['config'])
    # read all datasets
    datasets_config = analysis_config['datasets']
    datasets = []

    for dataset in datasets_config:
        dataset_config = datasets_config[dataset]
        data_provider = DataSetProvider(dataset_config)
        data_set = data_provider.get_dataset()
        datasets.append(data_set)

    as_plot = AlphasRunningPlot(datasets)
    as_plot.do_plot()


def min_func(asmz, metadataset):
    """Function to be minimized"""
    for dataset in metadataset.datasets:
        dataset.get_source(label='theory').set_asmz(asmz)
    chi2 = get_chi2(metadataset.get_data(),
                    metadataset.get_theory(),
                    metadataset.get_cov_matrix())

    print "asmz", asmz
    print "chi2", chi2
    print metadataset.get_nbins()
    return chi2


def get_chi2(data, theory, cov_matrix):
    """Simple definition to calculate chi2 using covariance matrix."""
    inv_matrix = np.matrix(cov_matrix).getI()
    residual = np.matrix(data - theory)
    chi2 = (residual * inv_matrix * residual.getT())[0, 0]
    return chi2


class TheoryCalculatorSource(Source):

    def __init__(self, asmz=0.1184, mz=91.18, nflavor=5, nloop=4, algo='crundec',
                 label=None, origin=None):
        super(TheoryCalculatorSource, self).__init__(None, label=label, origin=origin)
        self._asmz = asmz
        self._mz = mz
        self._nflavor = nflavor
        self._nloop = nloop
        self._algo = algo
        self._qarr = None
        self._calc_asqarr = np.vectorize(self._calc_asq)

    def prepare(self, dataset):
        qarr = dataset.get_source('q').get_arr()
        self._qarr = qarr

    def set_qarr(self, qarr):
        self._qarr = qarr

    def set_asmz(self, asmz):
        self._asmz = asmz

    def set_mz(self, mz):
        self._mz = mz

    def set_nflavor(self, nflavor):
        self._nflavor = nflavor

    def set_nloop(self, nloop):
        self._nloop = nloop

    def _calc_asq(self, q):
        crundec = CRunDec()
        asq = crundec.AlphasExact(self._asmz, self._mz, q, self._nflavor, self._nloop)
        return asq

    def get_arr(self):
        return self._calc_asqarr(self._qarr)


