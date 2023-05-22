import numpy as np
from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_fake_utils import FakeObservations, config, add_option
import sn_script_input
import pandas as pd
import copy

# this is to load option for fake cadence
path = sn_script_input.__path__
confDict_fake = make_dict_from_config(path[0], 'config_cadence.txt')


parser = OptionParser(description='Script to generate fake scenarios.')

add_option(parser, confDict_fake)

#opts, args = parser.parse_args()

# parser = OptionParser()

parser.add_option('--configFile', type='str', default='input/DESC_cohesive_strategy/scenario_1.csv',
                  help='config file to use[%default]')

opts, args = parser.parse_args()

config_fake = config(confDict_fake, opts)
configFile = opts.configFile

# load configs
df_conf = pd.read_csv(configFile, comment='#')

df_conf['seasons'] = df_conf['seasons'].astype(str)
df_conf['seasonLength'] = df_conf['seasonLength'].astype(str)
print(df_conf)

config_fake = config(confDict_fake, opts)

# generate fake obs

cols = df_conf.columns
for i, row in df_conf.iterrows():
    for cc in cols:
        if 'Nvisits' not in cc:
            config_fake[cc] = row[cc]
        else:
            b = cc.split('_')[-1]
            config_fake['Nvisits'][b] = row[cc]
    print(config_fake)
    fakeData = FakeObservations(config_fake).obs
    print('fake Data', len(fakeData))
