# import os
from iminuit import Minuit


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

    def _min_func(self, asmz):
        """Function to be minimized.

           Here a basically the chi2 of the datasets is minimized by taking into account all correlations
           between different datasets and within the datasets themselves.
        """
        # self._dataset.set_theory_parameters(alphasmz=kwargs['asmz'])
        self._dataset.set_theory_parameters(alphasmz=asmz)
        return self._dataset.get_chi2()

    def do_fit(self):
        # Chi2 tolerance for error evaluation
        pars = dict(asmz=0.118)
        m = Minuit(self._min_func, **pars)
        #m.print_param()
        m.migrad()
        self.values = m.values
        self.errors = m.errors
        # self._calc_par_uncert()

