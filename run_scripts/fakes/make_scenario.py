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
parser.add_option("--m5_file", type=str,
                  default='input/m5_OS/m5_field_night_baseline_v3.0_10yrs.csv',
                  help="file to process [%default]")
parser.add_option("--airmass_max", type=float,
                  default=1.8,
                  help="max airmass for obs (for fast run_mode only) [%default]")

opts, args = parser.parse_args()

config_fake = config(confDict_fake, opts)
configFile = opts.configFile
m5_file = opts.m5_file
airmass_max = opts.airmass_max

# get m5 values
m5 = pd.read_csv(m5_file)
idx = m5['airmass'] <= airmass_max
m5 = m5[idx]

m5_med = m5.groupby(['note', 'season', 'filter'])[
    'fiveSigmaDepth', 'airmass'].median().reset_index()

print(m5_med)
# load configs
df_conf = pd.read_csv(configFile, comment='#')

df_conf['seasons'] = df_conf['seasons'].astype(str)
df_conf['seasonLength'] = df_conf['seasonLength'].astype(str)
print(df_conf)

config_fake = config(confDict_fake, opts)

# generate fake obs

cols = df_conf.columns

fakeData = None
for i, row in df_conf.iterrows():
    for cc in cols:
        if 'Nvisits' not in cc:
            config_fake[cc] = row[cc]
        else:
            b = cc.split('_')[-1]
            config_fake['Nvisits'][b] = row[cc]
    # update m5 values
    seasonb = list(map(int, row['seasons']))
    idx = m5_med['note'] == row['field']
    idxb = m5_med['season'].isin(seasonb)
    selm5 = m5_med[idx & idxb]
    for b in selm5['filter'].unique():
        io = selm5['filter'] == b
        rr = selm5[io]
        config_fake['m5'][b] = str(rr['fiveSigmaDepth'].values[0])

    print(config_fake)
    fakes = FakeObservations(config_fake).obs
    if fakeData is None:
        fakeData = np.copy(fakes)
    else:
        fakeData = np.concatenate((fakeData, fakes))

print('fake Data', len(fakeData), fakeData)