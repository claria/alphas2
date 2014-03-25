import numpy
import fastnlo
from fastnlo import Alphas


class MetaDataSet(object):
    """Holds all datasets and returns global covariance matrix
       and theory/ data arrays
       """
    def __init__(self):

        self._datasets = []
        self._nbins = 0

    def update_theory(self, asmz):
        for dataset in self._datasets:
            qvals = dataset.get_source(label='q').get_arr()
            theory = TheoryProvider(asmz)
            dataset.theory = theory.get_theory()

    def get_theory(self, func, *args, **kwargs):
        return numpy.concatenate([dataset.get_theory(func, *args, **kwargs) for dataset in self._datasets])

    def get_data(self):
        return numpy.concatenate([dataset.get_data() for dataset in self._datasets])

    def get_cov_matrix(self):
        cov_matrix = numpy.zeros((self._nbins,self._nbins))
        i = 0
        for dataset in self._datasets:
            dataset_cov_matrix = dataset.get_cov_matrix()
            for j in range(0,dataset._nbins):
                for k in range(0,dataset._nbins):
                    cov_matrix[i+j][i+k] = dataset_cov_matrix[j][k]
            i = dataset._nbins
        return cov_matrix

    def add_dataset(self, dataset):
        self._datasets.append(dataset)
        self._nbins += dataset._nbins

class DataSet(object):
    """Describes one dataset in the fit. There are two types of sources for this
       Source: data, theory, bins, correction_factors
       UncertaintySource: experimental and theoretical uncertainties
    """
    def __init__(self, sources, data=None, theory=None):

        self._theory = theory
        self._data = data
        self._uncertainties = {}
        self._bins = {}
        self._nbins = None
        self.scenario = 'all'

        if sources is not None:
            self.add_sources(sources)

    #
    #Theory
    #
    def set_theory(self, source):
        self._theory = source
        if self._nbins is None:
            self._nbins = source.arr.size
        else:
            if not self._nbins == source.arr.size:
                raise Exception('Mismatch of nbins of added source {}.'.format(source.label))

    def get_theory(self, func, *args, **kwargs):
        """return theory predictions"""
        qvals = self.get_source(label='q').get_arr()
        theory = numpy.array([func(q, *args, **kwargs) for q in qvals])

        #theory = self._theory.get_arr(*args,**kwargs)
        return theory

    theory = property(get_theory, set_theory)
    #
    #Data
    #
    def set_data(self, source):
        self._data = source
        if source is not None:
            self._nbins = source.arr.size

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
            if source.origin in ['theo_uncert', 'exp_uncert']:
                if (source .label not in self.scenario) and \
                        (self.scenario != "all"):
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
        cov_matrix = numpy.zeros((self._nbins, self._nbins))

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

        diag_uncert = numpy.zeros((2, self._nbins))
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
            diag_uncert += numpy.square(uncertainty.get_arr(symmetric=False))

        return numpy.sqrt(diag_uncert)

    def get_source(self, label):
        """Return Observable of given label"""
        if label == 'data':
            return self.data
        elif label == 'theory':
            return self.theory
        elif label in self._bins:
            return self._bins[label]
        elif label in self._uncertainties:
            return self._uncertainties[label]
        else:
            raise Exception('Label not found in sources.')


class Source(object):
    """
    Describes sources of various quantities like data, theory,
    correction factors or uncertainties
    array is a numpy array of the quantity
    label can be arbitrarily set to identify source
    origin identifies the type of source
      data, theory: identify theo measurement, prediction quantity
      correction: are additional correction factors applied to prediction
      bin: defines the phasespace of the measurement
      exp, theo: are experimental, theoretical sources of uncertainty
    """
    def __init__(self, arr, label=None, origin=None, mask=None):

        self._label = None
        self.label = label

        self._nbins = None
        self._arr = None
        if arr is not None:
            self.arr = arr

        self._origin = None
        self.origin = origin

        self._mask = None
        if mask is None:
            self.mask = (arr == arr)
        else:
            self._mask = mask

    def __str__(self):
        return self._label

    def __repr__(self):
        return self._label

    def __call__(self, *args, **kwargs):
        return self.arr

    #
    # mask
    #
    def set_mask(self, mask):
        """
        Mask specicific values (bins) which will ignored.
        """
        self._mask = mask
        self.nbins = numpy.count_nonzero(mask)

    def get_mask(self):
        return self._mask

    mask = property(get_mask, set_mask)

    def reset_mask(self):
        self.mask = (self._array == self._array)

    def masked(self, arr):
        #if self._mask is None:
        #    return arr
        if arr.ndim == 2:
            size = numpy.count_nonzero(self._mask)
            if arr.shape[0] == arr.shape[1]:
                mask = numpy.outer(self._mask, self._mask)
                return arr[mask].reshape((size, size))
            elif arr.shape[0] == 2:
                mask = numpy.vstack((self._mask, self._mask))
                return arr[mask].reshape((2, size))
        elif arr.ndim == 1:
            return arr[self._mask]
        else:
            raise Exception("Ndim {} not matched".format(arr.ndim))

    def set_nbins(self, nbins):
        self._nbins = nbins

    def get_nbins(self):
        return self._nbins

    nbins = property(get_nbins, set_nbins)

    def set_arr(self, arr):
        if not arr.ndim == 1:
            raise Exception("Source Dimension must be 1.")
        self._arr = arr
        self.nbins = arr.shape

    def get_arr(self):
        return self.masked(self._arr)

    arr = property(get_arr, set_arr)

    def set_origin(self, origin):
        if origin not in ['data', 'theory', 'data_correction', 'theo_correction',
                          'bin', 'exp_uncert', 'theo_uncert']:
            raise Exception("{} is not a valid origin.".format(origin))
        self._origin = origin

    def get_origin(self):
        return self._origin

    origin = property(get_origin, set_origin)

    def set_label(self, label):
        self._label = label

    def get_label(self):
        return self._label

    label = property(get_label, set_label)

class CalculatorSource(Source):

    def __init__(self, func, label=None, origin=None, mask=None):
        super(CalculatorSource, self).__init__(arr=None, label=label, origin=origin, mask=mask)
        self._func = func

    def get_arr(self, *args, **kwargs):
        return func(*args, **kwargs)


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
                 corr_matrix=None, corr_type=None, error_scaling='none', mask=None):

        super(UncertaintySource, self).__init__(arr=arr, label=label, origin=origin, mask=mask)

        #Either covariance matrix or diagonal elements must be provided.
        if arr is None and cov_matrix is None:
            raise Exception('Either diagonal uncertainty or covariance matrix must be provided.')

        if cov_matrix is None:
            if arr.ndim == 1:
                self._symmetric = True
                self._nbins = arr.shape[0]
                self._arr = numpy.vstack((arr, arr))
            elif arr.ndim == 2:
                self._symmetric = False
                self._nbins = arr.shape[1]
                self._arr = arr
        else:
            self._symmetric = True
            arr = numpy.sqrt(cov_matrix.diagonal())
            self._nbins = arr.shape[0]
            self._arr = numpy.vstack((arr, arr))
            self._corr_matrix = None
            self.corr_matrix = cov_matrix / numpy.outer(self._arr[0], self._arr[0])
        if cov_matrix is None and corr_matrix is not None:
            self._corr_matrix = None
            self.corr_matrix = corr_matrix

        self._corr_type = corr_type
        self.error_scaling = error_scaling

        #Check if source was already scaled
        self._chk_scaled = False

        # Set Diagonal error elements

        self._mask = self._arr[0] == self._arr[0]

    def __call__(self, symmetric=False, *args, **kwargs):
        return self.get_arr(symmetric=True)

    def set_arr(self, arr):
        if arr.ndim == 1:
            # Symmetric uncertainty
            self._arr = numpy.array((arr, arr))
        elif arr.ndim == 2:
            # Assume it is asymmetric
            self._arr = arr
        self._nbins = self._arr.shape[1]

    def get_arr(self, symmetric=False):
        if symmetric is True:
            return self.masked(self._symmetrize(self._arr))
        else:
            return self.masked(self._arr)

    # def set_relative_uncertainty(self, rel_uncert, ref, symmetric=False):
    #     self._abs_uncert = rel_uncert * ref
    #     self._nbins = self._abs_uncert.shape[1]
    #     self._symmetric = symmetric
    #
    # def get_relative_uncertainty(self, ref, symmetric=False):
    #     if symmetric == True:
    #         return self._sym_uncert / ref
    #     else:
    #         return self._abs_uncert / ref

    def set_correlation_matrix(self, corr_matrix):
        self._corr_type = 'bintobin'
        self._corr_matrix = corr_matrix

    def get_correlation_matrix(self):
        if self._corr_type == 'corr':
            return self.masked(numpy.ones((self._nbins, self._nbins)))
        elif self._corr_type == 'uncorr':
            return self.masked(numpy.identity(self._nbins))
        elif self._corr_type == 'bintobin':
            return self.masked(self._corr_matrix)
        else:
            raise Exception('Correlation Type invalid.')

    corr_matrix = property(get_correlation_matrix, set_correlation_matrix)

    def scale(self, theory=None, data=None):
        if self._chk_scaled is True:
            raise Exception('Source already scaled.')
        if self.error_scaling == 'none':
            pass
        elif self.error_scaling == 'additive':
            #TODO: check that theo. sources do not scale!
            pass
        elif self.error_scaling == 'multiplicative':
            #TODO: check that only experimental sources are rescaled
            self._arr *= (numpy.abs(theory / data))
        elif self.error_scaling == 'poisson':
            raise Exception('Not yet implemented.')
        else:
            raise Exception('No valid error scaling')

        self._chk_scaled = True

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
        arr = numpy.sqrt(cov_matrix.diagonal())
        self._corr_matrix = cov_matrix / numpy.outer(arr, arr)
        self.set_arr(arr)
        self._symmetric = True

    def get_cov_matrix(self):
        """
        Get covariance matrix
        """
        if self._corr_type == 'corr':
            return (numpy.outer(
                self.get_arr(symmetric=True),
                self.get_arr(symmetric=True)))
        elif self._corr_type == 'uncorr':
            return numpy.diagflat(
                numpy.square(self.get_arr(symmetric=True)))
        elif self._corr_type == 'bintobin':
            return (numpy.outer(
                self.get_arr(symmetric=True),
                self.get_arr(symmetric=True)) *
                self.get_correlation_matrix())
        else:
            raise Exception('Correlation type not valid.')

    cov_matrix = property(get_cov_matrix, set_cov_matrix)

    @staticmethod
    def _symmetrize(arr):
        """One possibility to symmetrize uncertainty of shape (2,xxx)"""
        return 0.5 * (arr[0] + arr[1])
