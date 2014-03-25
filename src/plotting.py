import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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
        pass

    def produce(self):
        # Pass all datasets + config
        # or just the config
        for dataset in self._datasets:
            x = dataset.get_source(label='q').get_arr()
            y = dataset.get_data()
            print "data"
            print zip(x, y)
            self.ax.scatter(x, y)

        # Plot theory prediction

    def finalize(self):
        self._save_fig()
        plt.close(self.fig)
