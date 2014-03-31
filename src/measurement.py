import numpy as np

from chi2 import Chi2Cov

try:
    from fastnlo import CRunDec
except ImportError:
    from fastnloreader import CRunDec


class GobalDataSet(object):
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


class DataSet(object):
    """Describes one dataset in the fit. There are two types of sources for this
       Source: data, theory, bins, correction_factors
       UncertaintySource: experimental and theoretical uncertainties
    """
    def __init__(self, sources=None, data=None, theory=None):

        # All member attributes
        self._theory = None
        self._data = None

        if theory is not None:
            self.theory = theory
        self.data = data
        self.scenario = 'all'
        self._uncertainties = {}
        self._bins = {}
        self._corrections = {}
        if sources is not None:
            self.add_sources(sources)

    #
    # nbins
    #
    def get_nbins(self):
        return self.data.size

    nbins = property(get_nbins)

    #
    #Theory
    #
    def set_theory(self, source):
        self._theory = source
        self._theory.prepare(self)

    def get_theory(self):
        """return theory predictions"""
        theory = self._theory.get_arr()
        return theory

    theory = property(get_theory, set_theory)

    def set_theory_parameters(self, **kwargs):
        self._theory.set_theory_parameters(**kwargs)

    #
    #Data
    #
    def set_data(self, source):
        self._data = source

    def get_data(self):
        return self._data.get_arr()

    data = property(get_data, set_data)

    def _add_source(self, source):

            if source.origin == 'data':
                self.data = source
            elif source.origin == 'theory':
                self.theory = source
            elif source.origin == 'bin':
                self._bins[source.label] = source
            elif source.origin in ['theo_correction', 'data_correction']:
                self._corrections[source.label] = source
            elif source.origin in ['theo_uncert', 'exp_uncert']:
                self._uncertainties[source.label] = source
            else:
                raise Exception('Source origin not known')

    def add_sources(self, sources):
        for source in sources:
            # TODO: Move to add_source
            if source.origin in ['theo_uncert', 'exp_uncert']:
                if (source .label not in self.scenario) and (self.scenario != "all"):
                    print "Omitting source {}".format(source)
                    continue
            self._add_source(source)

    def has_uncert(self, corr_type=None, origin=None, label=None):
        for uncertainty in self._uncertainties:
            if corr_type is not None:
                if uncertainty.corr_type in corr_type:
                    return True
            if origin is not None:
                if uncertainty.origin in origin:
                    return True
            if label is not None:
                if uncertainty.label in label:
                    return True
        return False

    def get_uncert_list(self, corr_type=None, origin=None, label=None):
        uncert_list = []
        for uncertainty in self._uncertainties.values():
            if corr_type is not None:
                if uncertainty.corr_type not in corr_type:
                    continue
            if origin is not None:
                if uncertainty.origin not in origin:
                    continue
            if label is not None:
                if uncertainty.label not in label:
                    continue
            uncert_list.append(uncertainty)
        return uncert_list

    def get_cov_matrix(self, corr_type=None, origin=None, label=None):
        cov_matrix = np.zeros((self.nbins, self.nbins))

        for uncertainty in self._uncertainties.values():
            if corr_type is not None:
                if uncertainty.corr_type not in corr_type:
                    continue
            if origin is not None:
                if uncertainty.origin not in origin:
                    continue
            if label is not None:
                if uncertainty.label not in label:
                    continue
            cov_matrix += uncertainty.get_cov_matrix()

        return cov_matrix

    def get_diagonal_unc(self, corr_type=None, origin=None, label=None):

        diag_uncert = np.zeros((2, self.nbins))
        for uncertainty in self._uncertainties.values():
            if corr_type is not None:
                if uncertainty.corr_type not in corr_type:
                    continue
            if origin is not None:
                if uncertainty.origin not in origin:
                    continue
            if label is not None:
                if uncertainty.label not in label:
                    continue
            diag_uncert += np.square(uncertainty.get_arr(symmetric=False))

        return np.sqrt(diag_uncert)

    def get_source(self, label):
        """Return Observable of given label"""
        if label == 'data':
            return self._data
        elif label == 'theory':
            return self._theory
        elif label in self._bins:
            return self._bins[label]
        elif label in self._uncertainties:
            return self._uncertainties[label]
        else:
            raise Exception('Label not found in sources.')

    def get_chi2(self):
        chi2_calculator = Chi2Cov(self)
        return chi2_calculator.get_chi2()


class Source(object):
    """
    Describes sources of various quantities like data, theory,
    correction factors or uncertainties
    array is a np array of the quantity
    label can be arbitrarily set to identify source
    origin identifies the type of source
      data, theory: identify theo measurement, prediction quantity
      correction: are additional correction factors applied to prediction
      bin: defines the phasespace of the measurement
      exp, theo: are experimental, theoretical sources of uncertainty
    """
    def __init__(self, arr, label=None, origin=None):

        # All member attributes
        self._arr = None
        self._origin = None
        self._label = None

        # Iniatialize member variables
        self._arr = arr
        self.label = label
        self.origin = origin

    def __str__(self):
        return self._label

    #
    # array
    #
    def get_arr(self):
        return self._arr

    def set_arr(self, arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        # if not arr.ndim == 1:
        #     raise Exception("Source Dimension must be 1.")
        self._arr = arr

    #
    # nbins
    #
    def get_nbins(self):
        return self._arr.size
    nbins = property(get_nbins)

    #
    # origin
    #
    def set_origin(self, origin):
        if origin not in ['data', 'theory', 'bin',
                          'data_correction', 'theo_correction',
                          'exp_uncert', 'theo_uncert']:
            raise Exception("{} is not a valid origin.".format(origin))
        self._origin = origin

    def get_origin(self):
        return self._origin

    origin = property(get_origin, set_origin)

    #
    # Label
    #
    def set_label(self, label):
        self._label = label

    def get_label(self):
        return self._label

    label = property(get_label, set_label)


class UncertaintySource(Source):
    """
    Based on Source:
    Additionally saves diagonal(uncorrelated) uncertainty and correlation matrix
    cov_matrix: Covariance matrix of uncertainty source
    corr_matrix: Correlation matrix of uncertainty source
    corr_type: Describes type of correlation [uncorr, corr, bintobin]
      corr: correlated uncertainty source
      uncorr: Uncorrelated source of uncertainty
      bintobin: Bin-to-bin correlations. Must be provided in correlation/covariance matrix
    error_scaling:
      additive: Source does not scale with truth value (fixed source)
      multiplicative: Source scales with truth value
      poisson: Assuming poisson-like scaling of uncertainty
    """
    def __init__(self, arr=None, label=None, origin=None, cov_matrix=None,
                 corr_matrix=None, corr_type=None):

        # All member attributes
        self._corr_matrix = None
        self._corr_type = None

        # Either covariance matrix or diagonal elements must be provided.
        if arr is None and cov_matrix is None:
            raise Exception('Either diagonal uncertainty or covariance matrix must be provided.')
        if arr is not None and cov_matrix is not None:
            raise Exception('Please provide either a diagonal uncertainty or a cov matrix, not both.')
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        # No covariance is provided
        if cov_matrix is None:
            if arr.ndim == 1:
                arr = np.vstack((arr, arr))
            elif arr.ndim == 2:
                pass
            else:
                raise Exception('Please provide a 1-dim or 2-dim array.')
        else:
            # Use the covariance matrix
            arr = np.sqrt(cov_matrix.diagonal())
            arr = np.vstack((arr, arr))
            self.corr_matrix = cov_matrix / np.outer(self._arr[0], self._arr[0])

        # Call super __init__ with constructed array
        super(UncertaintySource, self).__init__(arr=arr, label=label, origin=origin)

        if cov_matrix is None and corr_matrix is not None:
            self.corr_matrix = corr_matrix

        self._corr_type = corr_type

    def get_arr(self, symmetric=True):
        if symmetric is True:
            return self._symmetrize()
        else:
            return self._arr

    #
    # nbins
    #
    def get_nbins(self):
        return self.get_arr().size

    nbins = property(get_nbins)

    #
    # correlation matrix
    #
    def set_correlation_matrix(self, corr_matrix):
        self._corr_type = 'bintobin'
        self._corr_matrix = corr_matrix

    def get_correlation_matrix(self):
        if self._corr_type == 'corr':
            return np.ones((self.nbins, self.nbins))
        elif self._corr_type == 'uncorr':
            return np.identity(self.nbins)
        elif self._corr_type == 'bintobin':
            return self._corr_matrix
        else:
            raise Exception('Correlation Type invalid.')

    corr_matrix = property(get_correlation_matrix, set_correlation_matrix)

    #
    # Correlation type
    #
    def set_corr_type(self, corr_type):
        self._corr_type = corr_type

    def get_corr_type(self):
        return self._corr_type

    corr_type = property(get_corr_type, set_corr_type)

    #
    # Covariance matrix
    #
    def set_cov_matrix(self, cov_matrix):
        """
        Set Covariance matrix
        """
        arr = np.sqrt(cov_matrix.diagonal())
        self._corr_matrix = cov_matrix / np.outer(arr, arr)
        self.set_arr(arr)

    def get_cov_matrix(self):
        """
        Get covariance matrix
        """
        if self._corr_type == 'corr':
            return np.outer(self.get_arr(), self.get_arr())
        elif self._corr_type == 'uncorr':
            return np.diagflat(np.square(self.get_arr()))
        elif self._corr_type == 'bintobin':
            return np.outer(self.get_arr(), self.get_arr()) * self.get_correlation_matrix()
        else:
            raise Exception('Correlation type not valid.')

    cov_matrix = property(get_cov_matrix, set_cov_matrix)

    #
    # Error symmetrization
    #
    def _symmetrize(self):
        """One possibility to symmetrize uncertainty of shape (2,xxx)"""
        return 0.5 * (self._arr[0] + self._arr[1])


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

    def set_theory_parameters(self, asmz=None, mz=None, nflavor=None, nloop=None):

        if asmz is not None:
            self._asmz = asmz
        if mz is not None:
            self._asmz = asmz
        if nflavor is not None:
            self._asmz = asmz
        if nloop is not None:
            self._asmz = asmz

    def _calc_asq(self, q):
        crundec = CRunDec()
        asq = crundec.AlphasExact(self._asmz, self._mz, q, self._nflavor, self._nloop)
        return asq

    def get_arr(self):
        return self._calc_asqarr(self._qarr)
