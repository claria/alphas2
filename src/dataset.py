import logging
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np

from chi2 import Chi2Cov
from fnlo import fastNLOUncertaintiesAlphas

logger = logging.getLogger(__name__)


class DataSetBase(object):
    __metaclass__ = ABCMeta
    """ Abstract dataset base class describing all neccessary methods.

        A valid dataset class derives from DatasetBase and implements all of it functions.
        These are functions neccessary for a chi2 fit using
    """

    @abstractmethod
    def get_nbins(self):
        return

    @abstractmethod
    def get_theory(self):
        return

    @abstractmethod
    def get_data(self):
        return

    @abstractmethod
    def get_cov_matrix(self):
        return

    @abstractmethod
    def get_cov_matrix(self):
        return

    @abstractmethod
    def get_chi2(self):
        return


class GobalDataSet(DataSetBase):
    """Holds all datasets and returns global covariance matrix
       and theory/ data arrays
       """
    def __init__(self):

        self._datasets = []
        self._nbins = 0

    def get_theory(self):
        return np.concatenate([dataset.get_theory() for dataset in self._datasets])

    theory = property(get_theory)

    def set_theory_parameters(self, **kwargs):
        for dataset in self._datasets:
            dataset.set_theory_parameters(**kwargs)

    def get_data(self):
        return np.concatenate([dataset.get_data() for dataset in self._datasets])

    data = property(get_data)

    def get_nbins(self):
        return self._nbins

    def get_cov_matrix(self):
        cov_matrix = np.zeros((self._nbins, self._nbins))
        i = 0
        for dataset in self._datasets:
            dataset_cov_matrix = dataset.get_cov_matrix()
            for j in range(0, dataset.nbins):
                for k in range(0, dataset.nbins):
                    cov_matrix[i+j][i+k] = dataset_cov_matrix[j][k]
            i += dataset.nbins
        return cov_matrix

    def add_dataset(self, dataset):
        self._datasets.append(dataset)
        self._nbins += dataset.nbins

    def get_chi2(self):
        chi2_calculator = Chi2Cov(self)
        return chi2_calculator.get_chi2()


class DataSet(DataSetBase):
    """ Collection of all neccesary sources to calculate chi2

        Describes one dataset in the fit. There are two types of sources for this
        Source: data, theory, bins, correction_factors
        UncertaintySource: experimental and theoretical uncertainties
    """
    def __init__(self, label, sources=None):

        # All member attributes
        if label is not None:
            self._label = label

        self._nbins = None
        self._theory = None
        self._data = None
        self._scenario = None

        self._uncertainties = {}
        self._bins = {}
        self._corrections = {}

        if sources is not None:
            self.add_sources(sources)

    #########
    # Nbins #
    #########

    def get_nbins(self):
        return self.data.size

    def set_nbins(self, nbins):
        if self._nbins is None:
            self._nbins = nbins

    nbins = property(get_nbins, set_nbins)

    #########
    # Label #
    #########

    def get_label(self):
        return self._label

    def set_label(self, label):
        self._label = label

    label = property(get_label, set_label)

    ##########
    # Theory #
    ##########

    def set_theory(self, source):
        self._theory = source

    def get_theory(self):
        """return theory predictions"""
        return self._get_corrected(self._theory)

    theory = property(get_theory, set_theory)

    ########
    # Data #
    ########

    def set_data(self, source):
        self._data = source

    def get_data(self):
        return self._get_corrected(self._data)

    data = property(get_data, set_data)

    def _add_source(self, source):
        #TODO: Check if source already exists and throw warning
            if source.source_type == 'data':
                self.data = source
            elif source.source_type == 'theory':
                self.theory = source
            elif source.source_type == 'bin':
                self._bins[source.label] = source
            elif source.source_type in ['theo_correction', 'data_correction']:
                self._corrections[source.label] = source
            elif source.source_type in ['theo_uncert', 'exp_uncert']:
                self._uncertainties[source.label] = source
            else:
                raise Exception('Source source_type not known')

    def add_sources(self, sources):
        for source in sources:
            self._add_source(source)

    def has_uncert(self, corr_type=None, source_type=None, label=None):
        for uncertainty in self._uncertainties:
            if corr_type is not None:
                if uncertainty.corr_type in corr_type:
                    return True
            if source_type is not None:
                if uncertainty.source_type in source_type:
                    return True
            if label is not None:
                if uncertainty.label in label:
                    return True
        return False

    def get_uncert_list(self, corr_type=None, source_type=None, label=None):
        uncert_list = []
        for uncertainty in self._uncertainties.values():
            if corr_type is not None:
                if uncertainty.corr_type not in corr_type:
                    continue
            if source_type is not None:
                if uncertainty.source_type not in source_type:
                    continue
            if label is not None:
                if uncertainty.label not in label:
                    continue
            uncert_list.append(self._get_scaled(uncertainty))
        return uncert_list

    def get_cov_matrix(self, corr_type=None, source_type=None, label=None):
        cov_matrix = np.zeros((self.nbins, self.nbins))
        for uncertainty in self._uncertainties.values():
            if corr_type is not None:
                if uncertainty.corr_type not in corr_type:
                    continue
            if source_type is not None:
                if uncertainty.source_type not in source_type:
                    continue
            if label is not None:
                if uncertainty.label not in label:
                    continue
            cov_matrix += self._get_scaled(uncertainty).get_cov_matrix()

        return cov_matrix

    def get_diagonal_unc(self, corr_type=None, source_type=None, label=None):

        diag_uncert = np.zeros((2, self.nbins))
        for uncertainty in self._uncertainties.values():
            if corr_type is not None:
                if uncertainty.corr_type not in corr_type:
                    continue
            if source_type is not None:
                if uncertainty.source_type not in source_type:
                    continue
            if label is not None:
                if uncertainty.label not in label:
                    continue
            diag_uncert += np.square(self._get_scaled(uncertainty).get_arr(symmetric=False))

        return np.sqrt(diag_uncert)

    def get_scaled_source(self, label):
        """Return copy of underlying source with all corrections or scalings applied.

        :param label: Name of the source
        :type label: str
        :returns: Object with label 'label'
        :rtype : src.sources.Source
        """
        if label == 'data':
            return self._get_corrected(self._data)
        elif label == 'theory':
            return self._get_corrected(self._theory)
        elif label in self._bins:
            return self._get_scaled(self._bins[label])
        elif label in self._uncertainties:
            return self._get_scaled(self._uncertainties[label])
        else:
            raise ValueError('Label not found in sources.')

    def get_raw_source(self, label):
        """Return raw underlying source, without any corrections or scalings applied.

        :param label: Name of the source
        :type label: str
        :returns: Object with label 'label'
        :rtype : src.sources.Source
        """
        if label == 'data':
            return self._data
        elif label == 'theory':
            return self._theory
        elif label in self._bins:
            return self._bins[label]
        elif label in self._uncertainties:
            return self._uncertainties[label]
        else:
            raise ValueError('Label not found in sources.')

    def _get_corrected(self, src):
        new_src = deepcopy(src)
        # Apply corrections
        if new_src.source_type not in ('data', 'theory'):
            raise ValueError('Only data or theory can be corrected at the moment.')
        for correction in self._corrections.values():
            if new_src.source_type == 'data' and correction.source_type == 'data_correction':
                new_src *= correction.get_arr()
            if new_src.source_type == 'theory' and correction.source_type == 'theo_correction':
                new_src *= correction.get_arr()

        return new_src

    def _get_scaled(self, src):
        """ Returns a scaled source

            Checks if src is relative or absolute and scales it to absolute if neccesary.
            If src is an uncertainty source, checks if the error_scaling is additive or multiplicative
            and applies the neccesary scalings.
        :param src: Raw sources which should be scaled
        :return: Returns copy of the raw source with all scalings/corrections applied.
        :rtype: src.sources.Source
        """
        new_src = src.copy()

        if new_src.source_type not in ('exp_uncert', 'theo_uncert',
                                       'theo_correction', 'data_correction'):
            raise ValueError("Only exp and theo uncertainties and corrections can be rescaled at the moment.")

        # Apply rescaling of relative uncertainties
        if new_src.source_relation == 'absolute':
            pass
        elif new_src.source_relation == 'relative':
            if new_src.source_type in ['theo_uncert', 'theo_correction']:
                new_src.scale(self.theory)
            if new_src.source_type in ['exp_uncert', 'data_correction']:
                new_src.scale(self.data)
        elif new_src.source_relation == 'percentage':
            if new_src.source_type in ['theo_uncert', 'theo_correction']:
                new_src.scale(self.theory/100.)
            if new_src.source_type in ['exp_uncert', 'data_correction']:
                new_src.scale(self.data/100.)
        else:
            raise ValueError('The source_relation \"{}\" of source \"{}\" is invalid or not yet implemented.'.format(new_src.source_relation, new_src.label))

        # Apply rescaling of uncertainty
        if new_src.error_scaling is None or new_src.error_scaling == 'none':
            pass
        elif new_src.error_scaling == 'additive':
            pass
        elif new_src.error_scaling == 'multiplicative':
            new_src.scale(self.theory/self.data)
        else:
            raise ValueError('The requested error scaling \"{}\" of source \"{}\" is invalid or not yet implemented.'.format(new_src.error_scaling, new_src.label))

        return new_src

    def get_chi2(self):
        chi2_calculator = Chi2Cov(self)
        return chi2_calculator.get_chi2()


class FastNLODataset(DataSet):

    def __init__(self, fastnlo_table, pdfset, label, sources=None):
        super(FastNLODataset, self).__init__(label, sources)
        self._fastnlo_table = fastnlo_table
        self._pdfset = pdfset

        self._mz = 91.1876
        self._alphasmz = 0.1180
        self._fnlo = fastNLOUncertaintiesAlphas(self._fastnlo_table, self._pdfset)
        self._calculate_theory()

    def set_theory_parameters(self, asmz=None, mz=None):
        if asmz is not None:
            self._alphasmz = asmz
            self._fnlo.set_alphasmz(asmz)
        if mz is not None:
            self._mz = mz
            self._fnlo.set_mz(mz)
        self._calculate_theory()

    def _calculate_theory(self):

        xsnlo = self._fnlo.get_central_crosssection()
        self.get_raw_source('theory').set_arr(xsnlo)
        # self._theory = Source(xsnlo, label='xsnlo', source_type='theory')

        # Overwrite source, if existing, with current calculation
        if 'pdf_uncert' in self._uncertainties.keys():
            print "update pdf uncert"
            cov_pdf_uncert = self._fnlo.get_pdf_cov_matrix()
            self.get_raw_source('pdf_uncert').set_arr(cov_pdf_uncert)
        if 'scale_uncert' in self._uncertainties.keys():
            scale_uncert = self._fnlo.get_scale_uncert()
            self.get_raw_source('scale_uncert').set_arr(scale_uncert)


#class TestDataset(DataSet):
#
#    def __init__(self, fastnlo_table, pdfset, label, sources=None):
#        super(TestDataset, self).__init__(label, sources)
#
#        self._mz = 91.1876
#        self._alphasmz = 1.
#        self._calculate_theory()
#
#    def set_theory_parameters(self, alphasmz=None):
#        if alphasmz is not None:
#            self._alphasmz = alphasmz
#        self._calculate_theory()
#
#    def _calculate_theory(self):
#
#        theory = np.array([1., 1., 1.]) * self._alphasmz
#        self.get_raw_source('theory').set_arr(theory)


