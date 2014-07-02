import os
import numpy as np
import StringIO

from measurement import Source, UncertaintySource
from measurement import FastNLODataset, TestDataset
from configobj import ConfigObj
import config
import logging
logger = logging.getLogger(__name__)


class DataProvider(object):

    def __init__(self, filename, global_config):
        self.sources = []
        self._arr_dict = None
        self._dataset_config = None
        self._global_config = global_config

        self._dataset_path = os.path.join('data/', filename)
        self._read_datafile()
        self._parse_arraydict()

    def get_dataset(self):
        fastnlo_table = os.path.join(config.table_dir, self._dataset_config['config']['theory_table'])
        pdfset = self._global_config['pdfset']
        return TestDataset(fastnlo_table, pdfset, sources=self.sources,
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

    @staticmethod
    def _parse_identifier(identifier):
        identifier_list = identifier.split(':')
        #out = {'source_type': None,
        #      'corr_type': 'uncorr',
        #       'error_scaling': None,
        #       'source_relation': 'absolute'}
        out = {}
        for item in identifier_list[:]:
            # source_type
            if item in ['bin', 'data', 'theory', 'theo_correction', 'data_correction', 'exp_uncert', 'theo_uncert']:
                out['source_type'] = item
                identifier_list.remove(item)
                continue
            # corr_type
            if (item in ['uncorr', 'bintobin', 'corr']) or item.startswith('corr'):
                out['corr_type'] = item
                identifier_list.remove(item)
                continue
            # error_scaling
            if item in ['additive', 'multiplicative', 'poisson']:
                out['error_scaling'] = item
                identifier_list.remove(item)
                continue
            if item in ['absolute', 'relative', 'percentage']:
                out['source_relation'] = item
                identifier_list.remove(item)
                continue

        if identifier_list:
            raise ValueError('Invalid identifiers: {}'.format(identifier_list))

        return out

    def _parse_arraydict(self):
        #for label, item in self._arr_dict.items():
        for label in self._dataset_config['data_description'].keys():

            # Default properties
            prop = {'source_type': None,
                    'corr_type': 'uncorr',
                    'error_scaling': None,
                    'source_relation': 'absolute'}
            try:
                identifiers = self._parse_identifier(self._dataset_config['data_description'][label])
            except ValueError as e:
                raise ValueError("Source {}: {}".format(label, e.message))
            prop.update(identifiers)

            source_type = prop['source_type']
            corr_type = prop['corr_type']
            error_scaling = prop['error_scaling']
            source_relation = prop['source_relation']

            # There are three possible ways to identify a source
            # 1. The supplied label matches one source in self._arr_dict
            # 2. 'cov_' + label matches a source in self._arr_dict
            # 3. label + '_lo' and label + '_up' match sources in self._arr_dict
            if label in self._arr_dict:
                quantity = self._arr_dict[label]
            elif ("{}_lo".format(label) in self._arr_dict) and \
                 ("{}_up".format(label) in self._arr_dict):
                quantity = np.vstack((self._arr_dict["{}_lo".format(label)],
                                      self._arr_dict["{}_up".format(label)]))
            elif "cov_{}".format(label) in self._arr_dict:
                quantity = self._arr_dict["cov_{}".format(label)]
            # Dummy quantity is used. Theory qunatity will be calculated later.
            elif prop['source_type'] in ['theory', 'theo_uncert']:
                quantity = None
            else:
                raise Exception("Requested source \"{}\" not found in datafile.".format(label))

            # Item is a source and no uncertainty
            if source_type in ['bin', 'data_correction', 'theo_correction', 'data', 'theory']:
                source = Source(label=label, arr=quantity, source_type=source_type)
                self.sources.append(source)
            # Item is a uncertainty source
            elif source_type in ['exp_uncert', 'theo_uncert']:
                if quantity is not None:
                    if "corr_{}".format(label) in self._arr_dict:
                        corr_matrix = self._arr_dict["corr_{}".format(label)]
                    else:
                        corr_matrix = None

                    uncertainty_source = UncertaintySource(source_type=source_type,
                                                           arr=quantity,
                                                           corr_matrix=corr_matrix,
                                                           label=label,
                                                           corr_type=corr_type,
                                                           error_scaling=error_scaling,
                                                           source_relation=source_relation)
                elif quantity is None and prop['source_type'] == 'theo_uncert':
                    logger.debug('Adding dummy source for \"{}\". Meant to be replaced later by calculation.'.format(label))
                    uncertainty_source = UncertaintySource(source_type=source_type,
                                                           arr=[0],
                                                           label=label,
                                                           corr_type=corr_type,
                                                           error_scaling=error_scaling,
                                                           source_relation=source_relation)

                else:
                    raise ValueError('Correlation type \"{}\" not known.'.format(corr_type))
                self.sources.append(uncertainty_source)
            else:
                raise ValueError("Source \"{}\" is of unknown source_type \"{}\".".format(label, source_type))
