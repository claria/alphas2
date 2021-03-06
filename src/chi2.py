import os
import numpy as np
from abc import ABCMeta, abstractmethod


class Chi2(object):
    __metaclass__ = ABCMeta

    def __init__(self, dataset=None):
        self._chi2 = 0.0
        self._dataset = dataset
        self._data = self._dataset.data
        self._theory = self._dataset.theory

    def get_chi2(self):
        self._calculate_chi2()
        return self._chi2

    @abstractmethod
    def _calculate_chi2(self):
        pass


class Chi2Cov(Chi2):
    """ Simple chi2 calculator weighting the residuum of data and theory with the inverse of the covariance matrix
        of all uncertainty sources found in the fit.
    """

    def __init__(self, dataset=None):
        super(Chi2Cov, self).__init__(dataset)

    def _calculate_chi2(self):
        inv_matrix = np.linalg.inv(self._dataset.get_cov_matrix())
        residual = self._data - self._theory
        self._chi2 = np.inner(np.inner(residual, inv_matrix), residual)


class Chi2Nuisance(Chi2):
    """Calculate Chi2 with different kinds of uncertainties.

    Uncorrelated uncertainties and partly correlated uncertanties are
    treated using a covariance matrix  while fully correlated
    uncertainties are treated using nuisance parameters.
    """
    def __init__(self, dataset):
        super(Chi2Nuisance, self).__init__(dataset)
        self._cov_matrix = self._dataset.get_cov_matrix(unc_treatment='cov')
        self._inv_matrix = np.linalg.inv(self._cov_matrix)

        self._chi2_correlated = 0.0
        self._theory_mod = None

        # Correlated Sources for which nuisance parameters need to be calculated analytically.
        self._beta = self._dataset.get_uncert_ndarray(unc_treatment='nuis')
        self._r = None
        # Correlated sources for which nuisance parameters will be provided by Minuit.
        self._beta_external = []
        self._r_external = []

    def set_external_nuisance_parameters(self, beta_external, r_external):
        self._beta_external = beta_external
        self._r_external = r_external

    def get_nuisance_parameters(self):
        """ Returns list of nuisance parameters
        :return:
        """
        return self._r

    def get_chi2_correlated(self):
        """ Return chi2 contribution by correlated sources (nuisance parameters)
        :return:
        """
        return self._chi2_correlated

    def get_theory_modified(self):
        """ Return theory modified by nuisance parameters.
        :return:
        """
        return self._theory_mod

    def get_npts(self):
        """  Return number of datapoints in fit
        :return:
        """
        return self._dataset.data.shape[0]

    def _calculate_chi2(self):

        chi2_corr = 0.0
        nbeta = len(self._beta)
        b = np.zeros((nbeta,))
        a = np.identity(nbeta)

        # Calculate Bk and Akk' according to paper: PhysRevD.65.014012
        # Implementation adapted to the one of the HERAFitter
        self._theory_mod = self._theory.copy()

        for k in range(0, len(self._beta_external)):
            self._theory_mod = self._theory_mod - self._r_external[k] * self._beta_external[k]
            chi2_corr += self._r_external[k] ** 2

       # TODO: Get rid of one loop by switching to ndarray of _beta
        for k in range(0, nbeta):
            b[k] = np.inner(np.inner(self._data - self._theory_mod, self._inv_matrix), self._beta[k])
            for j in range(0, nbeta):
                a[k, j] += np.inner(np.inner(self._beta[j], self._inv_matrix), self._beta[k])

        # Multiply by -1 so nuisance parameters correspond to shift
        # noinspection PyTypeChecker
        self._r = np.linalg.solve(a, b) * (-1.)
        # Calculate theory prediction shifted by nuisance parameters
        for k in range(0, nbeta):
            self._theory_mod = self._theory_mod - self._r[k] * self._beta[k]
            chi2_corr += self._r[k] ** 2
        # Add also external nuisance parameter, which are fitted by Minuit

        residual_mod = self._data - self._theory_mod
        self._chi2 = np.inner(np.inner(residual_mod, self._inv_matrix), residual_mod)
        self._chi2 += chi2_corr
