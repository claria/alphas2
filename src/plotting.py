import os
import numpy as np
import matplotlib.pyplot as plt

from measurement import TheoryCalculatorSource
from unilibs.baseplot import BasePlot


class AlphasRunningPlot(BasePlot):

    def __init__(self, datasets=None, **kwargs):
        super(AlphasRunningPlot, self).__init__(**kwargs)
        self.ax = self.fig.add_subplot(111)
        self._datasets = datasets
        self.prepare()
        self.set_style(self.ax, style='cmsprel')

    def prepare(self):
        self.ax.set_xlabel('$Q$ (GeV)', x=1.0, ha='right', size='large')
        self.ax.set_ylabel(r'$\alpha_\mathrm{{S}}(Q)$ ', y=1.0, ha='right', size='large')
        self.ax.set_xscale('log')
        pass

    def produce(self):
        # Pass all datasets + config
        # or just the config

        # Plot theory prediction
        # read in fit _results
        theory_path = os.path.join('output/', 'result_asmz.txt')
        asmz_result = np.genfromtxt(theory_path, names=True)
        theory = TheoryCalculatorSource(label='theory', origin='theory')
        theory.set_asmz(float(asmz_result['asq']))
        theory_qarr = np.linspace(10, 1200, 100)
        theory.set_qarr(theory_qarr)
        self.ax.plot(theory_qarr, theory.get_arr(), color='yellow')
        theory.set_asmz(asmz_result['asq'] + asmz_result['tot_h'])
        self.ax.plot(theory_qarr, theory.get_arr(), color='green')
        theory.set_asmz(asmz_result['asq'] - asmz_result['tot_l'])
        self.ax.plot(theory_qarr, theory.get_arr(), color='green')

        for dataset in self._datasets:
            x = dataset.get_source(label='q').get_arr()
            y = dataset.get_data()
            yerr = dataset.get_diagonal_unc(origin='exp_uncert')
            self.ax.errorbar(x=x, y=y, yerr=yerr, fmt='o')
        #Text fields
        self.ax.text(s=r'$\alpha_S (m_Z) = {} \pm xx$'.format(asmz_result['asq']),
                     x=1., y=1., ha='right', va='top', transform=self.ax.transAxes)

    def finalize(self):

        self.ax.set_xlim(10., 2000.)
        self.ax.set_ylim(0.05, 0.2)

        self._save_fig()
        plt.close(self.fig)


class ProfileLikelihoodPlot(BasePlot):

    def __init__(self, data=None, **kwargs):
        super(ProfileLikelihoodPlot, self).__init__(**kwargs)
        self.ax = self.fig.add_subplot(111)
        self._data = data
        self.prepare()
        self.set_style(self.ax, style='cmsprel')

    def prepare(self):
        self.ax.set_xlabel(r'$\alpha_\mathrm{{S}}(M_\mathrm{{Z}})$', x=1.0, ha='right', size='large')
        self.ax.set_ylabel('$\chi^2$', y=1.0, ha='right', size='large')
        self.ax.set_xscale('linear')

    def produce(self):
        # Pass all datasets + config
        # or just the config

        # Plot theory prediction
        # read in fit _results
        self.ax.plot(self._data['x'], self._data['y'], color='black')
        self.ax.axhline(y=min(self._data['y'])+1.)

    def finalize(self):

        self.ax.set_xlim(0.114, 0.118)
        min_data = min(self._data['y'])
        self.ax.set_ylim(min_data - 1., min_data + 10.)
        self._save_fig()
        plt.close(self.fig)




