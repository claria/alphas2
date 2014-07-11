from copy import deepcopy
import numpy as np


class Source(object):
    """ Source containing arbitrary quantities like bin edges, bins, data or theory

    Describes sources of various quantities like data, theory,
    correction factors or uncertainties

    Attributes:
        arr: numpy.ndarray
            Array containing the quantity of interest
        label: str
            Name, identifier of the source
        nbins: int
            Number of bins of quantity
        source_type: str
            source_type of the source describing the type of the quantity. Possible
            choices are [data, theory, bin, data_correction, theo_correction, exp_uncert, theo_uncert]
            data, theory:
                identify data cross section, or theory prediction
            data_correction, theo_correction:
                additional correction factors applied to theory prediction or data
            bin:
                Source describing the phasespace of the measurement
            exp_uncert, theo_uncert:
                experimental or theoretical uncertainty source
    """
    def __init__(self, arr=None, label=None, source_type=None, source_relation='absolute'):

        # All member attributes
        self._arr = None
        self._source_type = None
        self._label = None
        # Is it a source_relation or absolute uncertainty source
        self._source_relation = source_relation
        # Initialize member variables
        self.arr = arr
        self.label = label
        self.source_type = source_type

    def __str__(self):
        return self._label

    def copy(self):
        """ Returns deepcopy of object
        :return: Copy of self object
        :rtype: Source
        """
        return deepcopy(self)

    def scale(self, arr):
        """Scale the source with arr
        :param arr:
        """
        self._arr *= arr

    #########
    # array #
    #########

    def __mul__(self, rhs):
        res = deepcopy(self)
        res._arr *= rhs
        return res

    def __div__(self, rhs):
        res = deepcopy(self)
        res._arr /= rhs
        return res

    def get_arr(self):
        return self._arr

    def set_arr(self, arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        # if not arr.ndim == 1:
        #     raise Exception("Source Dimension must be 1.")
        self._arr = arr

    arr = property(get_arr, set_arr)

    #########
    # nbins #
    #########

    def get_nbins(self):
        return self._arr.size
    nbins = property(get_nbins)

    ###############
    # source_type #
    ###############

    def set_source_type(self, source_type):
        if source_type not in ['data', 'theory', 'bin',
                               'data_correction', 'theo_correction',
                               'exp_uncert', 'theo_uncert']:
            raise Exception("{} is not a valid source_type.".format(source_type))
        self._source_type = source_type

    def get_source_type(self):
        return self._source_type

    source_type = property(get_source_type, set_source_type)

    #########
    # Label #
    #########

    def set_label(self, label):
        self._label = label

    def get_label(self):
        return self._label

    label = property(get_label, set_label)

    ########################
    # relation to quantity #
    ########################

    def get_source_relation(self):
        return self._source_relation

    def set_source_relation(self, source_relation):
        self._source_relation = source_relation

    source_relation = property(get_source_relation, set_source_relation)


class UncertaintySource(Source):
    """ Uncertainty Source with all its properties

    Contains asymmetric or symmetric uncertainties. Stores the 1d/2d array of the diagonal elements of the covariance
    matrix and the correlation matrix. This allows to also keep asymmetric uncertainties.

    Attributes:
        corr_matrix: str
            Correlation matrix of uncertainty source
        corr_type: str
            Describes type of correlation [uncorr, corr, bintobin]
               corr: correlated uncertainty source
               uncorr: Uncorrelated source of uncertainty
               bintobin: Bin-to-bin correlations. Must be provided in correlation/covariance matrix
        error_scaling: str, optional
            Information how this Source should be scaled by the dataset [additive, multiplicative, poissonlike]
                additive: Source does not scale with truth. If source_type is exp_uncert, no rescaling is required, if
                          source_type is theo_uncert, then the source is rescaled to data.
                multiplicative: Source scales with truth value, If source_type is exp_uncert, data is rescaled to the
                                truth
                                value (theory). If source_type is theo_uncert, source is not rescaled.
                poisson: Assuming poisson-like scaling of uncertainty
        relative: bool, optional
            The Uncertainty Source is relative to the quantity of source_type, e.g. exp_uncert, theo_uncert.
    """
    def __init__(self, arr=None, label=None, source_type=None,
                 corr_matrix=None, corr_type='uncorr', source_relation='absolute', error_scaling=None):
        # possible cases
        # 1. cov given, no corr_type
        # 1. cov given, corr type
        # 2. arr given,  corr_type
        # 3. arr given,  corr_matrix
        super(UncertaintySource, self).__init__(label=label, source_type=source_type, source_relation=source_relation)

        print "#################"
        print label
        print corr_type
        print corr_matrix

        # All member attributes
        self._corr_matrix = None
        self._corr_type = None

        self._error_scaling = error_scaling

        # Convert arr into numpy array if not None
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        self.set_arr(arr)

        if corr_type is None and corr_matrix is None:
            raise ValueError(('Neither a covariance matrix nor a correlation matrix" '
                              ' or corr_type is provided for source {}.').format(label))

        # No covariance matrix given, but correlation matrix given
        if corr_type is not None and corr_type != 'bintobin':
            self.set_corr_matrix(corr_type=corr_type)
        elif corr_matrix:
            self.set_corr_matrix(corr_matrix)
        elif (corr_type == 'bintobin') and (corr_matrix is None):
            if self._is_covmatrix(arr):
                diag = arr.diagonal()
                corr_matrix = arr / np.outer(diag, diag)
                corr_matrix[np.isnan(corr_matrix)] = 0.
                self.set_corr_matrix(corr_matrix)
        else:
            raise Exception('Source {}: Either corr_matrix or corr_type must be provided.'.format(self._label))

    #######
    # arr #
    #######

    @staticmethod
    def _is_covmatrix(arr):
        if (arr.ndim == 2) and np.array_equal(arr.transpose(), arr):
            return True
        return False

    def set_arr(self, arr):
        # Either covariance matrix or diagonal elements must be provided.
        if arr is None:
            raise Exception(('Either diagonal uncertainty or covariance matrix'
                             'must be provided for source {}.').format(self._label))

        # TODO: Check if arr is a 1-dim source, a-dim asymmetric source or a nxn cov_matrix
        if arr.ndim == 1:
            arr = np.vstack((-1. * arr, arr))
        elif arr.ndim == 2:
            #Check if arr is symmetric --> means covariance matrix
            if self._is_covmatrix(arr):
                # TODO: Catch divison by zero if diagonal elements are zero
                diag = np.sqrt(arr.diagonal())
                corr_matrix = arr / np.outer(diag, diag)
                corr_matrix[np.isnan(corr_matrix)] = 0.
                self.set_corr_matrix(corr_matrix)

                arr = np.vstack((-1. * diag, diag))
            else:
                # Already have a asymmetric 2d source.
                pass
        else:
            raise Exception('A 1-dim or 2-dim array must be provided.')

        self._arr = arr

    def get_arr(self, symmetric=True):
        if not self._arr.ndim == 2:
            raise ValueError("Uncertainty source diagonal elements need to be 2d.")
        if symmetric is True:
            return 0.5 * (self._arr[1] - self._arr[0])
        else:
            return self._arr

    #########
    # nbins #
    #########

    def get_nbins(self):
        return self.get_arr().size

    nbins = property(get_nbins)

   #################
    # error scaling #
    #################

    def get_error_scaling(self):
        return self._error_scaling

    def set_error_scaling(self, error_scaling):
        if error_scaling not in ['additive', 'multiplicative', 'poisson']:
            raise ValueError('{} is not a valid error scaling model.'.format(error_scaling))
        self._error_scaling = error_scaling

    error_scaling = property(get_error_scaling, set_error_scaling)

    ######################
    # correlation matrix #
    ######################

    def set_corr_matrix(self, corr_matrix=None, corr_type=None):
        if corr_matrix is not None:
            self._corr_matrix = corr_matrix
        elif corr_type is not None:
            if corr_type == 'uncorr':
                self._corr_matrix = np.identity(self.nbins)
            elif corr_type.startswith('corr'):
                if len(corr_type.split('_')) == 2:
                    corr_factor = float(corr_type.split('_')[1])
                elif corr_type == 'corr':
                    corr_factor = 1.0
                else:
                    raise Exception('Invalid corr_type. {}'.format(corr_type))
                self._corr_matrix = (np.identity(self.nbins) +
                                    (np.ones((self.nbins, self.nbins)) -
                                     np.identity(self.nbins)) * corr_factor)
            elif corr_type == 'bintobin':
                pass
            else:
                raise Exception('Correlation type invalid.')
        else:
            raise Exception('Either corr_matrix or corr_type must be provided.')

    def get_corr_matrix(self):
        return self._corr_matrix

    corr_matrix = property(get_corr_matrix, set_corr_matrix)

    ####################
    # Correlation type #
    ####################

    def set_corr_type(self, corr_type):
        # TODO: stub function. should manipulate the correlation matrix using the given corr_type
        pass

    def get_corr_type(self):
        """Calculates and returns the corr_type of the covariance matrix. Needed for the nuisance parameter
           method in which fully correlated uncertainties are treated using nuisance parameters.
        """
        if np.array_equal(self.corr_matrix, np.identity(self.nbins)):
            return 'uncorr'
        elif np.array_equal(self.corr_matrix, np.ones((self.nbins, self.nbins))):
            return 'corr'
        else:
            return 'bintobin'

    corr_type = property(get_corr_type, set_corr_type)

    #####################
    # Covariance matrix #
    #####################

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
        return np.outer(self.get_arr(), self.get_arr()) * self.corr_matrix

    cov_matrix = property(get_cov_matrix, set_cov_matrix)