import os
from iminuit import Minuit
from scipy.optimize import root


class AlphasFitter(object):

    def __init__(self):
        pass

    def _min_func(self, *args):
        pass

    def do_fit(self):
        pass


class MinuitFitter(AlphasFitter):

    def __init__(self, dataset):
        super(MinuitFitter, self).__init__()
        self._fit_pars = None
        self._asmz_fit_uncert = None
        self._tolerance = 1.0
        self._bounds = (0.100, 0.200)

        self._dataset = dataset

    def _min_func(self, asmz, *args):
        """Function to be minimized.
           Here a basically the chi2 of the datasets is minimized by taking into account all correlations
           between different datasets and within the datasets themselves.
        """
        print asmz, self._dataset.get_chi2()
        self._dataset.set_theory_parameters(alphasmz=asmz)
        return self._dataset.get_chi2()

    # @staticmethod
    # def _root_func(pars, dataset, fitted_pars, tolerance=1.0):
    #     """ Function used to evaluate the parameter uncertainty. min_func(x) is shifted downwards using
    #         the f(best_fit) and the tolerance. The roots of this function are evaluated and define the
    #         parameter uncertainty using the given tolerance.
    #     """
    #     print "pars", pars
    #     chi2 = (AlphasFitter._min_func(pars, dataset) -
    #             AlphasFitter._min_func(fitted_pars, dataset) -
    #             tolerance)
    #     return chi2

    def do_fit(self):
        # Chi2 tolerance for error evaluation
        # fit_result = minimize(AlphasFitter._min_func,
        #                       [0.117],
        #                       args=(self._dataset,),
        #                       bounds=(self._bounds,),
        #                       options={'disp': True, 'gtol': 1e-6})
        m = Minuit(self._min_func, asmz=0.118, error_asmz=0.001, print_level=1)
        #m.print_param()
        m.migrad()
        print m.values
        print m.errors
        m.minos()
        print m.errors
        self._fit_pars = [0.]
        # self._calc_par_uncert()
        # Find root of function min_func - asmz_central + tolerance

    #     def _calc_par_uncert(self):
    #         asmz_l = root(AlphasFitter._root_func, [0.110],
    #                       args=(self._dataset, self._fit_pars, self._tolerance))
    #         print "roots", asmz_l.x
    #         # asmz_h = root(AlphasFitter._root_func, self._asmz_fit, self._bounds[1],
    #         #                 args=(self._dataset, self._asmz_fit, self._tolerance))
    #         #
    #         # self._asmz_fit_uncert = (self._asmz_fit - asmz_l, asmz_h - self._asmz_fit)

    def save_result(self, filepath=None):
        if filepath is None:
            filepath = os.path.join('output/', 'Result.txt')
        with open(filepath, 'w') as f:
            f.write('# {} {} {} {}\n'.format('q', 'asq', 'tot_l', 'tot_h'))
            f.write('{} {} {} {}'.format(91.18, self._asmz_fit, *self._asmz_fit_uncert))