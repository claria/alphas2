import os
import numpy as np
import StringIO

from measurement import Source, UncertaintySource
from measurement import DataSet
from config import config
from configobj import ConfigObj


class DataSetProvider(object):
    def __init__(self, filename):

        self.sources = []
        self._array_dict = None
        self._dataset_config = None

        self._dataset_path = os.path.join(config.data_dir, filename)
        self._read_datafile()

        self._parse_arraydict()


    def get_dataset(self):
        return DataSet(self.sources)

    def _read_datafile(self):
        #Split into two file objects
        configfile = StringIO.StringIO()
        datafile = StringIO.StringIO()
        with open(self._dataset_path) as f:
            file_input = f.readlines()

        config_part = True
        for line in file_input:
            if '[data]' in line:
                config_part = False
                continue
            if config_part:
                configfile.write(line)
            else:
                datafile.write(line)
        configfile.seek(0)
        datafile.seek(0)

        config = ConfigObj(configfile)['config']
        data =  np.genfromtxt(datafile, names=True)

        configfile.close()
        datafile.close()

        arr_dict = dict()
        for i in data.dtype.names:
            if data[i].ndim == 1:
                arr_dict[i] = np.array(data[i])
            elif data[i].ndim == 0:
                arr_dict[i] = np.array([data[i]])

        self._dataset_config = config
        self._array_dict = arr_dict

    def _parse_arraydict(self):
        #for label, item in self._array_dict.items():
        for label in self._dataset_config['description']:
            # Symmetric uncertainty source
            if label in self._array_dict:
                item = self._array_dict[label]
            # Asymmetric uncertainty source
            elif ("{}_l".format(label) in self._array_dict) and \
                 ("{}_h".format(label) in self._array_dict):
                item = np.vstack((self._array_dict["{}_l".format(label)],
                                  self._array_dict["{}_h".format(label)]))
            else:
                raise Exception("Requested source not found in datafile.")

            origin = self._dataset_config['description'][label]
            if origin in ['bin', 'data_correction', 'theo_correction', 'data', 'theory']:
                source = Source(label=label, arr=item, origin=origin)
                self.sources.append(source)
            elif origin in ['exp_uncert', 'theo_uncert']:
                corr_type = self._dataset_config['corr_type'][label]
                if corr_type in ['corr', 'uncorr']:
                    uncertainty_source = UncertaintySource(origin=origin,
                                                           arr=item,
                                                           label=label,
                                                           corr_type=corr_type)
                elif corr_type == 'bintobin':
                    if 'cov_' + label in self._array_dict:
                        uncertainty_source = UncertaintySource(origin=origin,
                                                               cov_matrix=self._array_dict[
                                                                   'cov_' + label],
                                                               label=label,
                                                               corr_type=corr_type)
                    elif 'cor_' + label in self._array_dict:
                        uncertainty_source = UncertaintySource(origin=origin,
                                                               arr=item,
                                                               corr_matrix=
                                                               self._array_dict[
                                                                   'cor_' + label],
                                                               label=label,
                                                               corr_type=corr_type)
                else:
                    raise Exception('Correlation type not known: {}'.format(corr_type))
                self.sources.append(uncertainty_source)
            else:
                print "Omitting unknown source {} of origin {}.".format(label, origin)

