import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from measurement import TheoryCalculatorSource
from config import config
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
        for dataset in self._datasets:
            x = dataset.get_source(label='q').get_arr()
            y = dataset.get_data()
            yerr = dataset.get_diagonal_unc(origin='exp_uncert')
            print "hola"
            print yerr
            self.ax.errorbar(x=x, y=y, yerr=yerr, fmt='o')

        # Plot theory prediction
        # read in fit _results
        theory_path = os.path.join(config.output_dir, 'result_asmz.txt')
        asmz_result = np.genfromtxt(theory_path, names=True)
        theory = TheoryCalculatorSource(label='theory', origin='theory')
        print asmz_result['asq']
        theory.set_asmz(float(asmz_result['asq']))
        theory_qarr = np.linspace(10, 1200, 100)
        print theory_qarr
        theory.set_qarr(theory_qarr)
        print theory.get_arr()
        self.ax.plot(theory_qarr, theory.get_arr(), color='yellow')

    def finalize(self):

        self.ax.set_xlim(10., 2000.)
        self.ax.set_ylim(0.05, 0.2)

        self._save_fig()
        plt.close(self.fig)
