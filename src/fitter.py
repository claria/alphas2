import os
import numpy

from measurement import DataSet, MetaDataSet
from providers import DataSetProvider
from config import config

from fastnlo import CRunDec

from scipy.optimize import minimize_scalar

def perform_fit(**kwargs):

    # read confi
    analysis_config = config.get_config(kwargs['config'])
    # read all datasets
    datasets_config = analysis_config['datasets']

    metadataset = MetaDataSet()

    for dataset in datasets_config:
        dataset_config = datasets_config[dataset]
        data_provider = DataSetProvider(dataset_config)
        data_set = data_provider.get_dataset()
        metadataset.add_dataset(data_set)


    # if there are correlations between the samples provide them
    pass

    # merge them all in one array + one covaraince matrix
    # calculate theory predictions for all data points
    # calculate chi2
    result = minimize_scalar(min_func,args=(metadataset,),
                             method='bounded', bounds=(0.1000,0.2000))
    print result

def get_alphasq(q, asmz=0.1184, mz=91.18, nflavor=5, nloop=4):
    crundec = CRunDec()
    asq = crundec.AlphasExact(asmz, mz, q, nflavor, nloop)
    return asq

def min_func(asmz, metadataset):
    """Function to be minimized"""
    # print metadataset.get_theory()
    # print metadataset.get_data()
    # print metadataset.get_cov_matrix()
    chi2 = get_chi2(metadataset.get_data(),
                    metadataset.get_theory(get_alphasq, asmz=asmz),
                    metadataset.get_cov_matrix())

    print "asmz", asmz
    print "chi2", chi2
    return chi2

def get_chi2(data, theory, cov_matrix):
    """Simple definition to calculate chi2 using covariance matrix."""
    inv_matrix = numpy.matrix(cov_matrix).getI()
    residual = numpy.matrix(data - theory)
    chi2 = (residual * inv_matrix * residual.getT())[0, 0]
    return chi2

def plot(**kwargs):
    pass
