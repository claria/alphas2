import os
import numpy as np
import StringIO

from measurement import Source, UncertaintySource
from measurement import FastNLODataset
from configobj import ConfigObj
import config
import logging
logger = logging.getLogger(__name__)


class DataProvider(object):

    def __init__(self, filename):
        super(DataProvider, self).__init__()
        self.sources = []
        self._arr_dict = None
        self._dataset_config = None

        self._dataset_path = os.path.join('data/', filename)
        self._read_datafile()
        self._parse_arraydict()

    def get_dataset(self):
        fastnlo_table = os.path.join(config.table_dir, self._dataset_config['config']['theory_table'])
        pdfset = 'CT10nlo.LHgrid'
        return FastNLODataset(fastnlo_table, pdfset, sources=self.sources,
                              label=self._dataset_config['config']['short_label'])

    def get_dataset_config(self):
        return self._dataset_config

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

        config = ConfigObj(configfile)
        data = np.genfromtxt(datafile, names=True)

        configfile.close()
        datafile.close()

        arr_dict = dict()
        for i in data.dtype.names:
            arr_dict[i] = np.atleast_1d(data[i])

        self._dataset_config = config
        self._arr_dict = arr_dict

    def _parse_arraydict(self):
        #for label, item in self._arr_dict.items():
        for label, origin in self._dataset_config['data_description'].items():
            # origin = self._dataset_config['data_description'][label]
            # Symmetric uncertainty source
            if label in self._arr_dict:
                item = self._arr_dict[label]
            # Asymmetric uncertainty source
            elif ("{}_l".format(label) in self._arr_dict) and \
                 ("{}_h".format(label) in self._arr_dict):
                item = np.vstack((self._arr_dict["{}_l".format(label)],
                                  self._arr_dict["{}_h".format(label)]))
            # Source can be added later on the fly. Dummy will be added for now.
            elif origin == 'theo_uncert':
                pass
            else:
                raise Exception("Requested source \"{}\" not found in datafile.".format(label))

            if origin in ['bin', 'data_correction', 'theo_correction', 'data', 'theory']:
                source = Source(label=label, arr=item, origin=origin)
                self.sources.append(source)
            elif origin in ['exp_uncert', 'theo_uncert']:
                if label in self._dataset_config['corr_type']:
                    corr_type = self._dataset_config['corr_type'][label]
                else:
                    logger.debug("No correlation type supplied for source \"{}\".".format(label))
                    corr_type = None

                if label in self._dataset_config['error_scaling']:
                    error_scaling = self._dataset_config['error_scaling'][label]
                else:
                    error_scaling = None

                if corr_type in ['corr', 'uncorr'] or corr_type.startswith('corr'):
                    uncertainty_source = UncertaintySource(origin=origin,
                                                           arr=item,
                                                           label=label,
                                                           corr_type=corr_type,
                                                           error_scaling=error_scaling)
                elif corr_type == 'bintobin':
                    print "dddd"
                    print label
                    if 'cov_' + label in self._arr_dict:
                        uncertainty_source = UncertaintySource(origin=origin,
                                                               cov_matrix=self._arr_dict['cov_' + label],
                                                               label=label,
                                                               corr_type=corr_type,
                                                               error_scaling=error_scaling)
                    elif 'cor_' + label in self._arr_dict:
                        uncertainty_source = UncertaintySource(origin=origin,
                                                               arr=item,
                                                               corr_matrix=self._arr_dict['cor_' + label],
                                                               label=label,
                                                               corr_type=corr_type,
                                                               error_scaling=error_scaling)
                    # Add dummy source, which needs to be overwritten later.
                    elif origin == 'theo_uncert':
                        uncertainty_source = UncertaintySource(origin=origin,
                                                               cov_matrix=np.atleast_2d(0.),
                                                               label=label,
                                                               corr_type=corr_type,
                                                               error_scaling=error_scaling)

                    else:
                        raise ValueError('No array or covariance matrix found for source \"{}\"'.format(label))
                else:
                    raise ValueError('Correlation type \"{}\" not known.'.format(corr_type))
                self.sources.append(uncertainty_source)
            else:
                raise ValueError("Source \"{}\" is of unknown origin \"{}\".".format(label, origin))
