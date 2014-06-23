import numpy as np
from chi2 import Chi2Cov
from unilibs.fnlo import fastNLOUncertainties
from copy import deepcopy


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
    # nbins #
    #########

    def get_nbins(self):
        return self.data.size

    def set_nbins(self, nbins):
        if self._nbins is None:
            self._nbins = nbins

    nbins = property(get_nbins, set_nbins)

    #########
    # label #
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
        return self._get_corrected(self._theory).get_arr()

    theory = property(get_theory, set_theory)

    ########
    # Data #
    ########

    def set_data(self, source):
        self._data = source

    def get_data(self):
        return self._get_corrected(self._data).get_arr()

    data = property(get_data, set_data)

    def _add_source(self, source):
        #TODO: Check if source already exists and throw warning
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
            uncert_list.append(self._get_scaled(uncertainty))
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
            cov_matrix += self._get_scaled(uncertainty).get_cov_matrix()

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
            diag_uncert += np.square(self._get_scaled(uncertainty).get_arr(symmetric=False))

        return np.sqrt(diag_uncert)

    def get_source(self, label):
        """Return Observable of given label"""
        if label == 'data':
            return self._get_corrected(self._data)
        elif label == 'theory':
            return self._get_corrected(self._theory)
        elif label in self._bins:
            return self._bins[label]
        elif label in self._uncertainties:
            return self._get_scaled(self._uncertainties[label])
        else:
            raise Exception('Label not found in sources.')

    def _get_corrected(self, src):
        new_src = deepcopy(src)
         # Apply corrections
        if new_src.origin not in ('data', 'theory'):
            raise ValueError('Only data or theory can be corrected at the moment.')
        for correction in self._corrections.values():
            if new_src.origin == 'data' and correction.origin == 'data_correction':
                new_src *= correction.get_arr()
            if new_src.origin == 'theory' and correction.origin == 'theo_correction':
                new_src *= correction.get_arr()

        return new_src

    def _get_scaled(self, src):
        new_src = deepcopy(src)
       # Apply rescaling of uncertainty
        if new_src.origin not in ('exp_uncert', 'theo_uncert'):
            raise ValueError("Only exp and theo uncertainties can be rescaled at the moment.")
        if new_src.error_scaling is None or new_src.error_scaling == 'none':
            pass
        elif new_src.error_scaling == 'additive':
            pass
        elif new_src.error_scaling == 'multiplicative':
            new_src *= self.theory/self.data
        else:
            raise ValueError('Invalid error scaling.')
        return new_src

    def get_chi2(self):
        chi2_calculator = Chi2Cov(self)
        return chi2_calculator.get_chi2()


class FastNLODataset(DataSet):

    def __init__(self, fastnlo_table, pdfset, label, sources=None):
        super(FastNLODataset, self).__init__(label, sources)
        self._fastnlo_table = fastnlo_table
        self._pdfset = pdfset

        self._alphasmz = 0.1180
        self._fnlo = fastNLOUncertainties(self._fastnlo_table, self._pdfset)
        self._calculate_theory()

    def set_theory_parameters(self):
        self._calculate_theory()

    def _calculate_theory(self):
        self._theory = Source(self._fnlo.get_central_crosssection(), label='xsnlo', origin='theory')
        # Overwrite source, if existing, with current calculation
        if "pdf_uncert" in self._uncertainties.keys():
            self._add_source(UncertaintySource(cov_matrix=self._fnlo.get_pdf_cov_matrix(),
                                               label='pdf_uncert',
                                               origin='theo_uncert'))
        # self._add_source(UncertaintySource(arr=self._fnlo.get_scale_uncert(),
        #                                    label='pdf_uncert',
        #                                    origin='theo_uncert',
        #                                    corr_type='corr'))


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
        origin: str
            Origin of the source describing the type of the quantity. Possible
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
    def __init__(self, arr, label=None, origin=None):

        # All member attributes
        self._arr = None
        self._origin = None
        self._label = None

        # Initialize member variables
        self.arr = arr
        self.label = label
        self.origin = origin

    def __str__(self):
        return self._label

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

    ##########
    # origin #
    ##########

    def set_origin(self, origin):
        if origin not in ['data', 'theory', 'bin',
                          'data_correction', 'theo_correction',
                          'exp_uncert', 'theo_uncert']:
            raise Exception("{} is not a valid origin.".format(origin))
        self._origin = origin

    def get_origin(self):
        return self._origin

    origin = property(get_origin, set_origin)

    #########
    # Label #
    #########

    def set_label(self, label):
        self._label = label

    def get_label(self):
        return self._label

    label = property(get_label, set_label)


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
                additive: Source does not scale with truth. If origin is exp_uncert, no rescaling is required, if
                          origin is theo_uncert, then the source is rescaled to data.
                multiplicative: Source scales with truth value, If origin is exp_uncert, data is rescaled to the truth
                                value (theory). If origin is theo_uncert, source is not rescaled.
                poisson: Assuming poisson-like scaling of uncertainty
        relative: bool, optional
            The Uncertainty Source is relative to the quantity of origin, e.g. exp_uncert, theo_uncert.
    """
    def __init__(self, arr=None, label=None, origin=None, cov_matrix=None,
                 corr_matrix=None, corr_type=None, relative=False, error_scaling=None):
        # possible cases
        # 1. cov given, no corr_type
        # 1. cov given, corr type
        # 2. arr given,  corr_type
        # 3. arr given,  corr_matrix

        # All member attributes
        self._corr_matrix = None
        self._corr_type = None
        # Is it a relative or absolute uncertainty source
        self._relative = relative
        self._error_scaling = error_scaling

        # Either covariance matrix or diagonal elements must be provided.
        if arr is None and cov_matrix is None:
            raise Exception(('Either diagonal uncertainty or covariance matrix'
                             'must be provided for source {}.').format(label))
        if arr is not None and cov_matrix is not None:
            raise Exception('Please provide either a diagonal uncertainty or a cov matrix, not both.')

        # No covariance is provided
        if cov_matrix is None:
           # Convert arr into numpy array if not None
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            if arr.ndim == 1:
                arr = np.vstack((arr, arr))
            elif arr.ndim == 2:
                pass
            else:
                raise Exception('A 1-dim or 2-dim array must be provided.')

            if corr_type is None and corr_matrix is None:
                raise ValueError(('Neither a covariance matrix nor a correlation matrix" '
                                 ' or corr_type is provided for source {}.').format(label))
        # Covariance matrix is provided
        else:
            # Extract diagonal elements from covariance matrix
            # TODO: Catch divison by zero if diagonal elements are zero
            # Throws a waring at the moment.
            arr = np.sqrt(cov_matrix.diagonal())
            arr = np.vstack((arr, arr))
            if corr_matrix is not None:
                raise Exception('No corr_matrix may be provided if a cov matrix is given.')
            corr_matrix = cov_matrix / np.outer(arr[0], arr[0])
            # A bit hacky...
            corr_matrix[np.isnan(corr_matrix)] = 0.
            if corr_type is None:
                corr_type = 'bintobin'

        # Call super __init__ with constructed array
        super(UncertaintySource, self).__init__(arr=arr, label=label, origin=origin)

        # No covariance matrix given, but correlation matrix given
        if corr_type is not None and corr_type != 'bintobin':
            self.set_correlation_matrix(corr_type=corr_type)
        elif corr_matrix is not None:
            self.set_correlation_matrix(corr_matrix=corr_matrix)
        else:
            raise Exception('Either corr_matrix or corr_type must be provided.')

    #######
    # arr #
    #######

    def get_arr(self, symmetric=True):
        if not self._arr.ndim == 2:
            raise ValueError("Uncertainty source diagonal elements need to be 2d.")
        if symmetric is True:
            return 0.5 * (self._arr[0] + self._arr[1])
        else:
            return self._arr

    #########
    # nbins #
    #########

    def get_nbins(self):
        return self.get_arr().size

    nbins = property(get_nbins)

    ########################
    # relative uncertainty #
    ########################

    def get_relative(self):
        return self._relative

    def set_relative(self, relative):
        self._relative = relative

    relative = property(get_relative, set_relative)

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

    def set_correlation_matrix(self, corr_matrix=None, corr_type=None):
        if corr_matrix is not None:
            self._corr_matrix = corr_matrix
        elif corr_type is not None:
            if corr_type == 'uncorr':
                self._corr_matrix = np.identity(self.nbins)
            elif corr_type.startswith('corr'):
                if len(corr_type.split(':')) == 2:
                    corr_factor = float(corr_type.split(':')[1])
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

    def get_correlation_matrix(self):
        return self._corr_matrix

    corr_matrix = property(get_correlation_matrix, set_correlation_matrix)

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
        if np.array_equal(self.corr_matrix,np.identity(self.nbins)):
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
