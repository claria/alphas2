import os

import matplotlib
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.libs.arraydict import ArrayDict
from config import config
from unilibs.baseplot import BasePlot


class AlphasSensitivityPlot(BasePlot):

    def __init__(self, measurements=None, **kwargs):
        super(AlphasSensitivityPlot, self).__init__(**kwargs)
        self.ax = self.fig.add_subplot(111)
        self.measurements = measurements
        self.prepare()
        self.set_style(self.ax, style='cmsprel')

    def prepare(self):
        pass

    def produce(self):
        pass

    def finalize(self):
        self._save_fig()
        plt.close(self.fig)
