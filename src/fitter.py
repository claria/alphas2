# import os
from iminuit import Minuit
from iminuit.util import make_func_code, describe
from chi2 import Chi2Nuisance


class AlphasFitter(object):

    def __init__(self):
        self.values = {}
        self.errors = {}

    def _min_func(self, *args):
        pass

    def do_fit(self):
        pass


class MinuitFitter(AlphasFitter):

    def __init__(self, dataset, user_initial_values=None):
        super(MinuitFitter, self).__init__()
        self._tolerance = 1.0
        self._dataset = dataset

        self._nuis_parameters_sources = self._dataset.get_uncert_list(unc_treatment=('fit',))
        nuis_parameters_fit = [uncertainty.label for uncertainty in self._nuis_parameters_sources]

        initial_values = {'asmz': 0.118, 'error_asmz': 0.0001, 'limit_asmz': (0.100, 2.)}
        for nuis_parameter in nuis_parameters_fit:
            initial_values[nuis_parameter] = 0.
            initial_values['error_{}'.format(nuis_parameter)] = 1.0
            initial_values['limit_{}'.format(nuis_parameter)] = None

        # Initial values can be overwritten by user
        if user_initial_values:
            initial_values.update(user_initial_values)

        pars = ['asmz'] + nuis_parameters_fit

        # Create a Minuit object with given starting values.
        self._m = Minuit(MinFunction(self._dataset, pars), **initial_values)

    def do_fit(self):
        # Chi2 tolerance for error evaluation
        # pars = dict(asmz=0.118, error_asmz=0.001, limit_asmz=(0.08, 0.1300))
        self._m.migrad()
        # self._m.minos()
        self.values = self._m.values
        self.errors = self._m.errors
        # self._calc_par_uncert()


class MinFunction:
    def __init__(self, dataset, pars):
        self._dataset = dataset
        self._pars = pars
        self._beta = self._dataset.get_uncert_list(unc_treatment=['fit'])
        self.func_code = make_func_code(pars)

    def default_errordef(self):
        return 1.0

    def __call__(self, *args):
        pars = dict(zip(self._pars, args))
        self._dataset.set_theory_parameters(asmz=pars['asmz'])
        # test
        nuis_pars = list(pars.keys())
        nuis_pars.remove('asmz')
        nuis_beta = self._beta
        nuis_r = [pars[label] for label in nuis_pars]
        self._chi2_calculator = Chi2Nuisance(self._dataset)
        self._chi2_calculator.set_external_nuisance_parameters(nuis_beta, nuis_r)
        chi2 = self._chi2_calculator.get_chi2()
        print chi2
        return chi2

