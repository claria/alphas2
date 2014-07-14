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

    def __init__(self, dataset):
        super(MinuitFitter, self).__init__()
        self._tolerance = 1.0
        self._dataset = dataset

        self._nuis_fit = self._dataset.get_uncert_list(unc_treatment=('fit',))
        pars = ['asmz'] + [uncertainty.label for uncertainty in self._nuis_fit]
        def_values = {'asmz': 0.1184}
        self._m = Minuit(MinFunction(self._dataset, pars, self._nuis_fit), **def_values)

    def _min_func(self, asmz):
        """Function to be minimized.

           Here a basically the chi2 of the datasets is minimized by taking into account all correlations
           between different datasets and within the datasets themselves.
        """
        self._dataset.set_theory_parameters(asmz=asmz)
        return self._dataset.get_chi2()

    def do_fit(self):
        # Chi2 tolerance for error evaluation
        # pars = dict(asmz=0.118, error_asmz=0.001, limit_asmz=(0.08, 0.1300))
        self._m.migrad()
        self.values = self._m.values
        self.errors = self._m.errors
        # self._calc_par_uncert()


class MinFunction:
    def __init__(self, dataset, pars, beta):
        self._dataset = dataset
        self._pars = pars
        self._beta = beta
        self.func_code = make_func_code(pars)

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
        print self._chi2_calculator.get_nuisance_parameters()
        return chi2

