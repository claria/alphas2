import os
from abc import ABCMeta, abstractmethod
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


class BasePlot(object):
    __metaclass__ = ABCMeta

    def __init__(self, output_fn='test', output_ext=('png',), style='none'):

        self.init_matplotlib()
        self.fig = plt.figure()

        self.output_fn = output_fn
        self.output_ext = output_ext
        self.style = style

    def do_plot(self):
        """
        Run all three plotting steps
        """
        self.prepare()
        self.produce()
        self.finalize()

    @abstractmethod
    def prepare(self, **kwargs):
        """
        Before plotting:
        Add axes to Figure, etc
        """
        pass

    @abstractmethod
    def produce(self):
        """
        Do the Plotting
        """
        pass

    @abstractmethod
    def finalize(self):
        """
        Apply final settings, autoscale etc
        Save the plot
        """
        self._save_fig()
        plt.close(self.fig)

    def _save_fig(self):
        """
        Save Fig to File and create directory structure
        if not yet existing.
        """
        #Check if directory exists and create if not
        directory = os.path.dirname(self.output_fn)

        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        for ext in self.output_ext:
            filename = "{}.{}".format(self.output_fn, ext)
            self.fig.savefig(filename, bbox_inches='tight')

    @staticmethod
    def init_matplotlib():
        """
        Initialize matplotlib with following rc
        """
        matplotlib.rcParams['lines.linewidth'] = 2
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.serif'] = 'lmodern'
        matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
        matplotlib.rcParams['font.monospace'] = 'Computer Modern Typewriter'
        matplotlib.rcParams['font.style'] = 'normal'
        matplotlib.rcParams['font.size'] = 20.
        matplotlib.rcParams['legend.fontsize'] = 14.
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rc('text.latex', preamble=r'\usepackage{helvet},\usepackage{sfmath}')
        # Axes
        matplotlib.rcParams['axes.linewidth'] = 2.
        matplotlib.rcParams['xtick.major.pad'] = 6.
        matplotlib.rcParams['axes.color_cycle'] = [(0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
                                                   (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
                                                   (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
                                                   (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
                                                   (1.0, 0.4980392156862745, 0.0),
                                                   (1.0, 1.0, 0.2),
                                                   (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
                                                   (0.9686274509803922, 0.5058823529411764, 0.7490196078431373),
                                                   (0.6, 0.6, 0.6)]
        # Saving
        matplotlib.rcParams['savefig.bbox'] = 'tight'
        matplotlib.rcParams['savefig.dpi'] = 150
        matplotlib.rcParams['savefig.format'] = 'png'

    #
    # Helper functions
    #
    @staticmethod
    def set_preset_text(ax, text, loc='topright', **kwargs):
        """
        Possible Positions : topleft, topright
        """
        if loc == 'topleft':
            kwargs.update({'x': 0.0, 'y': 1.01, 'va': 'bottom',
                           'ha': 'left'})
        elif loc == 'topright':
            kwargs.update({'x': 1.0, 'y': 1.01, 'va': 'bottom',
                           'ha': 'right'})
        else:
            raise Exception('Unknown loc.')

        ax.text(s=text, transform=ax.transAxes, color='Black', **kwargs)

    def set_style(self, ax, style, show_cme=False):
        """
        Some preset styles
        """
        if style == 'none':
            pass
        elif style == 'cmsprel':
            self.set_preset_text(ax, r"\textbf{CMS Preliminary}", loc='topleft')
            if show_cme:
                self.set_preset_text(ax, r"$\sqrt{s} = 7\/ \mathrm{TeV}$",
                                     loc='topleft', )
        else:
            self.set_preset_text(ax, r"\textbf{CMS}", loc='topleft')
            if show_cme:
                self.set_preset_text(ax, r"$\sqrt{s} = 7\/ \mathrm{TeV}$",
                                     loc='topleft', )

    @staticmethod
    def autoscale(ax, xmargin=0.0, ymargin=0.0, margin=0.0):
        # User defined autoscale with margins
        x0, x1 = tuple(ax.dataLim.intervalx)
        if margin > 0:
            xmargin = margin
            ymargin = margin
        if xmargin > 0:
            if ax.get_xscale() == 'linear':
                delta = (x1 - x0) * xmargin
                x0 -= delta
                x1 += delta
            else:
                delta = (x1 / x0) ** xmargin
                x0 /= delta
                x1 *= delta
            ax.set_xlim(x0, x1)
        y0, y1 = tuple(ax.dataLim.intervaly)
        if ymargin > 0:
            if ax.get_yscale() == 'linear':
                delta = (y1 - y0) * ymargin
                y0 -= delta
                y1 += delta
            else:
                delta = (y1 / y0) ** ymargin
                y0 /= delta
                y1 *= delta
            ax.set_ylim(y0, y1)

    @staticmethod
    def log_locator_filter(x, pos):
        """
        Add minor tick labels in log plots at 2* and 5*
        """
        s = str(int(x))
        if len(s) == 4:
            return ''
        if s[0] in ('2', '5'):
            return s
        return ''

    @staticmethod
    def steppify_bin(arr, isx=False):
        """
        Produce stepped array of arr, also of x
        """
        if isx:
            newarr = np.array(zip(arr[0], arr[1])).ravel()
        else:
            newarr = np.array(zip(arr, arr)).ravel()
        return newarr

    @staticmethod
    def set(obj, *args, **kwargs):
        """
        Apply Settings in kwargs, while defaults are set
        """
        funcvals = []
        for i in range(0, len(args) - 1, 2):
            funcvals.append((args[i], args[i + 1]))
        funcvals.extend(kwargs.items())
        for s, val in funcvals:
            attr = getattr(obj, s)
            if callable(attr):
                attr(val)
            else:
                setattr(obj, attr, val)


class GenericPlot(BasePlot):
    """
    Very simple generic plotting script
    A list of datasets has to be provided.
    A dataset is a dict with x and y keys and dx,dy
    """
    def __init__(self, datasets,
                 output_fn='test.png',
                 output_ext=('png', ),
                 props=None,
                 **kwargs):
        super(GenericPlot, self).__init__(output_fn=output_fn,
                                          output_ext=output_ext,
                                          **kwargs)

        self.output_fn = output_fn
        self.datasets = datasets

        self.ax = self.fig.add_subplot(111)
        self.props = props if props else {}
        self.props.update(kwargs.get('post_props', {}))
        self.pre_props = kwargs.get('pre_props', {})

    def prepare(self, **kwargs):

        for artist, props in self.props.items():
            obj = getattr(self, artist)
            self.set(obj, **props)

    def produce(self):
        for dataset in self.datasets:
            plot_type = dataset.get('plot_type', 'plot')
            if plot_type == 'plot':
                self.ax.plot(dataset['x'],
                             dataset['y'],
                             label=dataset.get('label', ''),
                             **dataset.get('props', {}))
            elif plot_type == 'errorbar':
                self.ax.errorbar(x=dataset['x'],
                                 xerr=dataset['dx'],
                                 y=dataset['y'],
                                 yerr=dataset['dy'],
                                 fmt='+',
                                 label=dataset.get('label', ''),
                                 **dataset.get('props', {}))
            elif plot_type == 'fill_between':
                self.ax.fill_between(x=dataset['x'],
                                     y1=dataset['y'] - dataset['dy'][0],
                                     y2=dataset['y'] + dataset['dy'],
                                     #label=dataset.get('label',''),
                                     **dataset.get('props', {}))
        self.ax.legend()

    def finalize(self):

        for artist, props in self.props.items():
            obj = getattr(self, artist)
            self.set(obj, **props)

        self.autoscale(self.ax, margin=0.1)
        self._save_fig()
        plt.close(self.fig)


def fill_between_steps(x, y1, y2=0, h_align='mid', ax=None, **kwargs):
    """ Fills a hole in matplotlib: Fill_between for step plots.

    Parameters :
    ------------

    x : array-like
        Array/vector of index values. These are assumed to be equally-spaced.
        If not, the result will probably look weird...
    y1 : array-like
        Array/vector of values to be filled under.
    y2 : array-Like
        Array/vector or bottom values for filled area. Default is 0.

    **kwargs will be passed to the matplotlib fill_between() function.

    """
    # If no Axes opject given, grab the current one:
    if ax is None:
        ax = plt.gca()
    # First, duplicate the x values
    if x.ndim == 1:
        xx = x.repeat(2)[1:]
        # Now: the average x binwidth
        xstep = (x[1:] - x[:-1]).mean()
        # Now: add one step at end of row.
        xx = np.append(xx, xx.max() + xstep)
        # Make it possible to change step alignment.
        if h_align == 'mid':
            xx -= xstep / 2.
        elif h_align == 'right':
            xx -= xstep
    else:
        xx = x.flatten(order='F')

    # Also, duplicate each y coordinate in both arrays
    y1 = y1.repeat(2)
    if type(y2) == np.ndarray:
        y2 = y2.repeat(2)

    # now to the plotting part:
    return ax.fill_between(xx, y1, y2=y2, **kwargs)



def plot_steps(x, y, h_align='mid', ax=None, **kwargs):
    """ Fills a hole in matplotlib: Fill_between for step plots.

    Parameters :
    ------------

    x : array-like
        Array/vector of index values. These are assumed to be equally-spaced.
        If not, the result will probably look weird...
    y1 : array-like
        Array/vector of values to be filled under.
    y2 : array-Like
        Array/vector or bottom values for filled area. Default is 0.

    **kwargs will be passed to the matplotlib fill_between() function.

    """
    # If no Axes opject given, grab the current one:
    if ax is None:
        ax = plt.gca()
    # First, duplicate the x values
    if x.ndim == 1:
        xx = x.repeat(2)[1:]
        # Now: the average x binwidth
        xstep = (x[1:] - x[:-1]).mean()
        # Now: add one step at end of row.
        xx = np.append(xx, xx.max() + xstep)
        # Make it possible to change step alignment.
        if h_align == 'mid':
            xx -= xstep / 2.
        elif h_align == 'right':
            xx -= xstep
    else:
        xx = x.flatten(order='F')

    # Also, duplicate each y coordinate in both arrays
    y = y.repeat(2)

    # now to the plotting part:
    return ax.plot(xx, y, **kwargs)


def ensure_latex(inp_str):
    """
    Return string with escaped latex incompatible characters.
    :param inp_str:
    :return:
    """
    chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\letteropenbrace{}',
        '}': r'\letterclosebrace{}',
        '~': r'\lettertilde{}',
        '^': r'\letterhat{}',
        '\\': r'\letterbackslash{}',
    }

    return ''.join([chars.get(char, char) for char in inp_str])