import os
import numpy as np
import StringIO

from dataset import FastNLODataset, TestDataset
from configobj import ConfigObj
import config
import logging
from src.sources import UncertaintySource, Source

logger = logging.getLogger(__name__)


class DataProvider(object):
    def __init__(self, filename, global_config):
        self.sources = []
        self._arr_dict = None
        self._dataset_config = None
        self._dataset = None
        self._global_config = global_config

        self._dataset_path = self.get_dataset_path(filename)
        self._dataset_config, self._arr_dict = self._read_datafile(self._dataset_path)

        # Check for additional files
        if 'additional_datafiles' in self._dataset_config['config']:
            for dataset_path in self._dataset_config['config'].as_list('additional_datafiles'):
                logger.debug('Reading additional file {}'.format(dataset_path))
                dataset_path = self.get_dataset_path(dataset_path)
                _, arr = self._read_datafile(dataset_path)
                self._arr_dict.update(arr)

        print self._arr_dict.keys()
        self._parse_arraydict()

    @staticmethod
    def get_dataset_path(filename):
        if os.path.exists(filename):
            dataset_path = filename
        elif os.path.exists(os.path.join('data/', filename)):
            dataset_path = os.path.join('data/', filename)
        else:
            raise Exception('Dataset file \"{}\" not found.'.format(filename))
        return dataset_path

    def prepare_dataset(self):
        if self._dataset_config['config']['theory_type'] == 'fastNLO':
            fastnlo_table = os.path.join(config.table_dir, self._dataset_config['config']['theory_table'])
            pdfset = self._global_config['pdfset']
            self._dataset = FastNLODataset(fastnlo_table, pdfset, sources=self.sources,
                                  label=self._dataset_config['config']['short_label'])
        # elif self._dataset_config['config']['theory_type'] == 'fastNLONormJets':
        #     fastnlo_table = os.path.join(config.table_dir, self._dataset_config['config']['theory_table'])
        #     pdfset = self._global_config['pdfset']
        #     self._dataset = FastNLODatasetNormJets(fastnlo_table, pdfset, sources=self.sources,
        #                           label=self._dataset_config['config']['short_label'])
        elif self._dataset_config['config']['theory_type'] == 'test':
            self._dataset = TestDataset(sources=self.sources, label=self._dataset_config['config']['short_label'])
        else:
            raise Exception('No valid theory_type specified for dataset \"{}\".'.format(
                self._dataset_config['config']['short_label']))

    def get_dataset(self):
        return self._dataset

    def get_dataset_config(self):
        return self._dataset_config

    @staticmethod
    def _read_datafile(dataset_path):
        # Split into two file objects
        configfile = StringIO.StringIO()
        datafile = StringIO.StringIO()
        with open(dataset_path) as f:
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

        dataset_config = ConfigObj(configfile)
        # noinspection PyTypeChecker

        arr_dict = dict()
        try:
            # noinspection PyTypeChecker
            data = np.genfromtxt(datafile, names=True)
            for i in data.dtype.names:
                arr_dict[i] = np.atleast_1d(data[i])

        except ValueError:
            # When the number of descriptors in the first comment line does not match the actual number of columns
            # a ValueError is rised. If only one descriptor is given, it is assumed that a 2d array is provided.
            datafile.seek(0)
            header = datafile.readline().lstrip('#').rstrip('\n')
            if not len(header.split()) == 1:
                raise ValueError('Currently only fully qualified field descriptors'
                                 'or single field descriptors ar allowed.')
            # noinspection PyTypeChecker
            data = np.genfromtxt(datafile)
            arr_dict[header] = np.atleast_1d(data)
        configfile.close()
        datafile.close()

        # self._dataset_config = dataset_config
        # self._arr_dict = arr_dict
        return dataset_config, arr_dict

    @staticmethod
    def _parse_identifier(identifier):
        # Remove leading and trailing colons and then split the string
        identifier_list = identifier.lstrip(':').rstrip(':').split(':')
        # out = {'source_type': None,
        # 'corr_type': 'uncorr',
        # 'error_scaling': None,
        #       'source_relation': 'absolute'}
        out = {}
        for item in identifier_list[:]:
            # source_type
            if item in ['bin', 'data', 'theory', 'theo_correction', 'data_correction', 'exp_uncert', 'theo_uncert']:
                out['source_type'] = item
                identifier_list.remove(item)
            # corr_type
            elif (item in ['uncorr', 'bintobin', 'corr']) or item.startswith('corr'):
                out['corr_type'] = item
                identifier_list.remove(item)
            # error_scaling
            elif item in ['additive', 'multiplicative', 'poisson']:
                out['error_scaling'] = item
                identifier_list.remove(item)
            elif item in ['absolute', 'relative', 'percentage']:
                out['source_relation'] = item
                identifier_list.remove(item)
            elif item in ['fit', 'cov', 'nuis']:
                out['unc_treatment'] = item
                identifier_list.remove(item)
            else:
                raise ValueError('Invalid identifiers: {}'.format(identifier_list))

        # Check that identifier list is empty. Probably not neccessary.
        if identifier_list:
            raise ValueError('Invalid identifiers: {}'.format(identifier_list))

        return out

    def _parse_arraydict(self):
        # for label, item in self._arr_dict.items():
        for label in self._dataset_config['data_description'].keys():

            # Default properties
            prop = {'source_type': None,
                    'corr_type': 'uncorr',
                    'error_scaling': None,
                    'source_relation': 'absolute',
                    'unc_treatment': 'cov'
                    }
            try:
                identifiers = self._parse_identifier(self._dataset_config['data_description'][label])
            except ValueError as e:
                raise ValueError("Source {}: {}".format(label, e.message))
            prop.update(identifiers)

            source_type = prop['source_type']
            corr_type = prop['corr_type']
            error_scaling = prop['error_scaling']
            source_relation = prop['source_relation']
            unc_treatment = prop['unc_treatment']

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
                                                           source_relation=source_relation,
                                                           unc_treatment=unc_treatment)
                elif quantity is None and prop['source_type'] == 'theo_uncert':
                    logger.debug(('Adding dummy source for \"{}\".'
                                  'Will be replaced later by fastNLO calculation.').format(label))
                    uncertainty_source = UncertaintySource(source_type=source_type,
                                                           arr=[0],
                                                           label=label,
                                                           corr_type=corr_type,
                                                           error_scaling=error_scaling,
                                                           source_relation=source_relation,
                                                           unc_treatment=unc_treatment)

                else:
                    raise ValueError('Correlation type \"{}\" not known.'.format(corr_type))
                self.sources.append(uncertainty_source)
            else:
                raise ValueError("Source \"{}\" is of unknown source_type \"{}\".".format(label, source_type))
