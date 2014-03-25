import os
import numpy as np

from measurement import Source, UncertaintySource
from measurement import DataSet
from config import config


class DataSetProvider(object):
    def __init__(self, dataset_config):

        self.sources = []
        self._array_dict = None

        self._dataset_config = dataset_config
        self._dataset_filename = dataset_config['datafilename']
        self._data_file = os.path.join(config.data_dir, self._dataset_filename)

        self._array_dict = self._read_datafile(self._data_file)
        print self._array_dict
        self._parse_arraydict()

    def get_dataset(self):
        return DataSet(self.sources)

    @staticmethod
    def _read_datafile(self, filename):
        arr = np.genfromtxt(filename, names=True)
        arr_dict = dict()
        for i in arr.dtype.names:
            if arr[i].ndim == 1:
                arr_dict[i] = np.array(arr[i])
            elif arr[i].ndim == 0:
                arr_dict[i] = np.array([arr[i]])
        return arr_dict

    def _parse_arraydict(self):
        for label, item in self._array_dict.items():
            if not label in self._dataset_config['description']:
                continue

            origin = self._dataset_config['description'][label]
            if origin in ['bin', 'data_correction', 'theo_correction', 'data', 'theory']:
                print "debug"
                print item
                print type(item)
                print item.size
                print item.ndim
                source = Source(label=label, arr=item, origin=origin)
                self.sources.append(source)
            elif origin in ['exp_uncert', 'theo_uncert']:
                corr_type = self._dataset_config['corr_type'][label]
                error_scaling = self._dataset_config['error_scaling'].get(label,
                                                                          'none')
                if corr_type in ['corr', 'uncorr']:
                    uncertainty_source = UncertaintySource(origin=origin,
                                                           arr=item,
                                                           label=label,
                                                           corr_type=corr_type,
                                                           error_scaling=error_scaling)
                elif corr_type == 'bintobin':
                    if 'cov_' + label in self._array_dict:
                        uncertainty_source = UncertaintySource(origin=origin,
                                                               cov_matrix=self._array_dict[
                                                                   'cov_' + label],
                                                               label=label,
                                                               corr_type=corr_type,
                                                               error_scaling=error_scaling)
                    elif 'cor_' + label in self._array_dict:
                        uncertainty_source = UncertaintySource(origin=origin,
                                                               arr=item,
                                                               corr_matrix=
                                                               self._array_dict[
                                                                   'cor_' + label],
                                                               label=label,
                                                               corr_type=corr_type,
                                                               error_scaling=error_scaling)
                else:
                    raise Exception('Correlation type not known: {}'.format(corr_type))
                self.sources.append(uncertainty_source)
            else:
                print "Omitting unknown source {} of origin {}.".format(label, origin)

