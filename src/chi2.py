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
        self._beta = self._dataset.get_uncert_list(unc_treatment='nuis')

        self._inv_matrix = np.linalg.inv(self._cov_matrix)

        self._chi2_correlated = 0.0
        self._theory_mod = None
        self._r = []
        self._beta_external = []
        self._r_external = []

    def set_external_nuisance_parameters(self, beta_external, r_external):
        self._beta_external = beta_external
        self._r_external = r_external
        pass

    def get_nuisance_parameters(self):
        beta_labels = [uncertainty.label for uncertainty in self._beta]
        nuis = dict(zip(beta_labels, self._r))
        beta_external_labels = [uncertainty.label for uncertainty in self._beta_external]
        nuis_external = dict(zip(beta_external_labels, self._r_external))
        nuis.update(nuis_external)
        return nuis

    def get_chi2_correlated(self):
        return self._chi2_correlated

    def get_theory_modified(self):
        return self._theory_mod

    def get_ndof(self):
        return self._dataset.data.shape[0]

    def save_nuisance_parameters(self, filename):
        nuisance_parameters = self.get_nuisance_parameters()
        # TODO: Move check to other part of code
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'w') as f:
            f.write("{:<20}: {:>10}\n".format("Nuis", "Shift in STD"))
            for key in sorted(nuisance_parameters.keys()):
                f.write("{:<20}: {:>10.2f}\n".format(
                    key, nuisance_parameters[key]))

    def _calculate_chi2(self):

        chi2_corr = 0.0
        nbeta = len(self._beta)
        b = np.zeros((nbeta,))
        a = np.identity(nbeta)

        # Calculate Bk and Akk' according to paper: PhysRevD.65.014012
        # Implementation adapted to the one of the HERAFitter
        # TODO: Get rid of one loop by switching to ndarray of _beta
        for k in range(0, nbeta):
            b[k] = (self._data - self._theory) * self._inv_matrix * self._beta[k].get_arr().T()
            for j in range(0, nbeta):
                a[k, j] += self._beta[j].get_arr() * self._inv_matrix * self._beta[k].get_arr().T()

        # Multiply by -1 so nuisance parameters correspond to shift
        # noinspection PyTypeChecker
        self._r = np.linalg.solve(a, b) * (-1.)
        # Calculate theory prediction shifted by nuisance parameters
        self._theory_mod = self._theory.copy()
        for k in range(0, nbeta):
            self._theory_mod = self._theory_mod - self._r[k] * self._beta[k].get_arr()
            chi2_corr += self._r[k] ** 2
        # Add also external nuisance parameter, which are fitted by Minuit
        for k in range(0, len(self._beta_external)):
            self._theory_mod = self._theory_mod - self._r_external[k] * self._beta_external[k].get_arr()
            chi2_corr += self._r_external[k] ** 2

        residual_mod = self._data - self._theory_mod
        self._chi2 = np.inner(np.inner(residual_mod, self._inv_matrix), residual_mod)
        self._chi2 += chi2_corr
