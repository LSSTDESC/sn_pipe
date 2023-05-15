#!/bin/bash
import os
import pandas as pd
from optparse import OptionParser


parser = OptionParser()

parser.add_option("--m5_file", type=str,
                  default='input/m5_OS/m5_field_season_baseline_v3.0_10yrs.csv',
                  help="file to process [%default]")
parser.add_option("--ref_field", type=str, default='DD:XMM_LSS',
                  help="ref field [%default]")
parser.add_option("--ref_season", type=int, default=3,
                  help="ref season [%default]")
opts, args = parser.parse_args()


m5_file = opts.m5_file
ref_field = opts.ref_field
ref_season = opts.ref_season

# get m5 values
m5 = pd.read_csv(m5_file)
"""
m5_med = m5.groupby(['note', 'filter'])[
    'fiveSigmaDepth', 'airmass'].median().reset_index()
"""
idx = m5['note'] == ref_field
idx &= m5['season'] == ref_season
m5_field = m5[idx]

print(m5_field)

cmd = 'python run_scripts/sim_to_fit/run_fake.py'
cmd += ' --SN_NSNabsolute=1 --MultiprocessingSimu_nproc=8'

for b in 'ugrizy':
    io = m5_field['filter'] == b
    sel = m5_field[io]
    cmd += ' --m5_{}={}'.format(b, sel['fiveSigmaDepth'].values[0])

print(cmd)
os.system(cmd)
