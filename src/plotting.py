import os
import matplotlib.pyplot as plt

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
        # Plot theory
        theory_path = os.path.join('output/', 'Result.txt')

        # Plot datasets
        for dataset in self._datasets:
            x = dataset.get_scaled_source(label='q').get_arr()
            y = dataset.get_data()
            yerr = dataset.get_diagonal_unc(origin='exp_uncert')
            self.ax.errorbar(x=x, y=y, yerr=yerr, fmt='o', label=dataset.label, zorder=10)

        #Get artists and labels for legend and chose which ones to display
        handles, labels = self.ax.get_legend_handles_labels()

        #Create custom artists
        fit_artist = plt.Line2D((0, 1), (0, 0), color='y')

        #Create legend from custom artist/label lists
        self.ax.legend([handle for handle in handles]+[fit_artist],
                       [label for label in labels]+['Alphas Fit'],
                       loc='upper right')

        #Text fields
        # self.ax.text(s=r'$\alpha_S (m_Z) = {} \pm xx$'.format(asmz_result['asq']),
        #              x=1., y=0.98, ha='right', va='top', transform=self.ax.transAxes)

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