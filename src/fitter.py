# import os
from iminuit import Minuit
from iminuit.util import make_func_code, describe
from chi2 import Chi2Nuisance


class MinuitFitter(object):

    def __init__(self, dataset, user_initial_values=None):
        super(MinuitFitter, self).__init__()
        self._tolerance = 1.0
        self._dataset = dataset

        self._nuis_parameters_sources = self._dataset.get_uncert_list(unc_treatment=('fit',))
        nuis_parameters_fit = [uncertainty.label for uncertainty in self._nuis_parameters_sources]

        initial_values = {'asmz': 0.118, 'error_asmz': 0.0001, 'limit_asmz': (0.100, 0.2)}
        for nuis_parameter in nuis_parameters_fit:
            initial_values[nuis_parameter] = 0.
            initial_values['error_{}'.format(nuis_parameter)] = 1.0
            initial_values['limit_{}'.format(nuis_parameter)] = None

        # Initial values can be overwritten by user
        if user_initial_values:
            initial_values.update(user_initial_values)

        # Always fit asmz (can be fixed) + nuisance parameters
        # Nuisance parameters which are not fitted are calculated analytically
        pars = ['asmz'] + nuis_parameters_fit

        # Create a Minuit object with given starting values.
        self._m = Minuit(MinFunction(self._dataset, pars), **initial_values)
        print "####### BEFORE FIT########"
        print "data"
        print self._dataset.get_data()
        print "theory"
        print self._dataset.get_theory()
        print "covmatrix"
        print self._dataset.get_cov_matrix()[0,0]
        print self._dataset.get_cov_matrix()[1,1]
        print self._dataset.get_cov_matrix()[1,0]
        print self._dataset.get_cov_matrix()[0,1]
        #import sys
        #sys.exit(0)

    def do_fit(self):
        # Chi2 tolerance for error evaluation
        # pars = dict(asmz=0.118, error_asmz=0.001, limit_asmz=(0.08, 0.1300))
        self._m.migrad()
        self._m.minos()

        print "####### AFTER FIT########"
        print "covmatrix"
        print self._m.fcn._dataset.get_cov_matrix()[0,0]
        print self._m.fcn._dataset.get_cov_matrix()[1,1]
        print self._m.fcn._dataset.get_cov_matrix()[1,0]
        print self._m.fcn._dataset.get_cov_matrix()[0,1]
         # Fit parameters and errors
        # print self._m.values
        # print self._m.errors
        # print self._m.get_fmin()

        # for dataset in self._dataset._datasets:
        #     print dataset.label
        #     self._chi2_calculator = Chi2Nuisance(dataset)
        #     print self._chi2_calculator.get_chi2()


        # print self._m.fcn._dataset.data
        # print self._m.fcn._dataset.theory

        # global chi2
        # chi2 per dataset
        # nuisance parameters
        # save parameter correlation matrix
        # extract nuisance parameters
        # extract chi2


class MinFunction:
    def __init__(self, dataset, pars):
        self._dataset = dataset
        self._pars = pars
        self._beta = self._dataset.get_uncert_ndarray(unc_treatment=['fit'])
        self.func_code = make_func_code(pars)
        self._nuisance_parameters = None

        self._fcn_calls = 0

    @staticmethod
    def default_errordef():
        """ Returns the default error definition used in iminuit.
        :return:
        """
        return 1.0

    def print_status(self, chi2, pars):
        print "Call {}: {}".format(self._fcn_calls, chi2)
        print "Params: {}".format(str(pars))

    def __call__(self, *args):
        """ Minimize function called by iminuit
        :param args: Fit parameters.
        :return: Calculated chi2 value.
        """
        pars = dict(zip(self._pars, args))
        self._dataset.set_theory_parameters(asmz=pars['asmz'])
        self._chi2_calculator = Chi2Nuisance(self._dataset)
        # test
        nuis_pars = list(pars.keys())
        nuis_pars.remove('asmz')
        nuis_beta = self._beta
        nuis_r = [pars[label] for label in nuis_pars]
        self._chi2_calculator.set_external_nuisance_parameters(nuis_beta, nuis_r)
        chi2 = self._chi2_calculator.get_chi2()
        # self._fcn_calls += 1
        self.print_status(chi2, pars)
        return chi2

