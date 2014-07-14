import os
import numpy as np
from abc import ABCMeta, abstractmethod


class Chi2(object):
    __metaclass__ = ABCMeta

    def __init__(self, measurement=None):
        self._chi2 = 0.0
        self._measurement = measurement
        self._data = self._measurement.data
        self._theory = self._measurement.theory

    def get_chi2(self):
        self._calculate_chi2()
        return self._chi2

    @abstractmethod
    def _calculate_chi2(self):
        pass


class Chi2Cov(Chi2):

    def __init__(self, measurement=None):
        super(Chi2Cov, self).__init__(measurement)

    def _calculate_chi2(self):
        inv_matrix = np.matrix(self._measurement.get_cov_matrix()).getI()
        residual = np.matrix(self._data - self._theory)
        self._chi2 = (residual * inv_matrix * residual.getT())[0, 0]


class Chi2Nuisance(Chi2):

    def __init__(self, measurement):
        super(Chi2Nuisance, self).__init__(measurement)
        self._cov_matrix = self._measurement.get_cov_matrix(unc_treatment=('cov',))
        self._inv_matrix = np.matrix(self._cov_matrix).getI()
        self._beta = self._measurement.get_uncert_list(unc_treatment=('nuis',))

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
        return self._measurement.data.shape[0]

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
        """Calculate Chi2 with different kinds of uncertainties.

        Uncorrelated uncertainties and partly correlated uncertanties are
        treated using a covariance matrix  while fully correlated
        uncertainties are treated using nuisance parameters.
        """

        chi2_corr = 0.0
        nbeta = len(self._beta)
        # npoints = len(data)
        b = np.zeros((nbeta,))
        a = np.matrix(np.identity(nbeta))
        # inv_matrix = cov_matrix.getI()

        # Calculate Bk and Akk' according to paper: PhysRevD.65.014012
        # Implementation similar to h1fitter
        for k in range(0, nbeta):
            b[k] = (np.matrix(self._data - self._theory)
                    * self._inv_matrix * np.matrix(self._beta[k].get_arr()).getT())
            # Better readable but much slower
            # for l in range(0,npoints):
            #    for i in range(0,npoints):
            #        B[k] += data[l] * syst_error[k][l]*
            #                (data[i]-theory[i]) * inv_matrix[l,i]
            for j in range(0, nbeta):
                a[k, j] += (np.matrix(self._beta[j].get_arr()) * self._inv_matrix *
                            np.matrix(self._beta[k].get_arr()).getT())
                # Better readable but way slower
                # for l in range(0,npoints):
                #    for i in range(0,npoints):
                #        A[k,j] += syst_error[k][i] * data[i] *
                #                 syst_error[j][l] * data[l] * inv_matrix[l,i]

        # Multiply by -1 so nuisance parameters correspond to shift
        # noinspection PyTypeChecker
        self._r = np.linalg.solve(a, b) * (-1)
        # Calculate theory prediction shifted by nuisance parameters
        self._theory_mod = self._theory.copy()
        for k in range(0, nbeta):
            self._theory_mod = self._theory_mod - self._r[k] * self._beta[k].get_arr()
            chi2_corr += self._r[k] ** 2
        for k in range(0, len(self._beta_external)):
            self._theory_mod = self._theory_mod - self._r_external[k] * self._beta_external[k].get_arr()
            chi2_corr += self._r_external[k] ** 2

        residual_mod = np.matrix(self._data - self._theory_mod)
        self._chi2 = (residual_mod * self._inv_matrix * residual_mod.getT())[
            0, 0]
        self._chi2 += chi2_corr
