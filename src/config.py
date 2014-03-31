import os
from ext.configobj import ConfigObj

config_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(config_dir, os.pardir))
table_dir = os.path.join(base_dir, 'tables')
data_dir = os.path.join(base_dir, 'data')
cache_dir = os.path.join(base_dir, 'cache')
cache_theory = os.path.join(cache_dir, 'theory')
cache_chi2 = os.path.join(cache_dir, 'chi2')
output_dir = os.path.join(base_dir, 'output')
output_plots = os.path.join(output_dir, 'plots')
output_nuisance = os.path.join(output_dir, 'nuisance')


def get_config(configname):
    return ConfigObj(os.path.join(config_dir, configname + '.conf'))
