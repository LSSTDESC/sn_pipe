#!/bin/bash
#!/bin/bash
import os
import pandas as pd
from optparse import OptionParser


parser = OptionParser()

parser.add_option("--m5_file", type=str,
                  default='input/m5_OS/m5_field_night_baseline_v3.0_10yrs.csv',
                  help="file to process [%default]")
parser.add_option("--seasons", type=str,
                  default='1',
                  help="seasons to process [%default]")
parser.add_option("--field", type=str,
                  default='DD:XMM_LSS',
                  help="field to process [%default]")
parser.add_option("--run_mode", type=str,
                  default='fast',
                  help="way of running [fast or slow]  [%default]")
parser.add_option("--airmass_max", type=float,
                  default=1.8,
                  help="max airmass for obs (for fast run_mode only) [%default]")
opts, args = parser.parse_args()


m5_file = opts.m5_file
seasons = opts.seasons
seasonb = seasons.split(',')
seasonb = list(map(int, seasonb))

field = opts.field
run_mode = opts.run_mode
airmass_max = opts.airmass_max

if run_mode == 'fast':
    # get m5 values
    m5 = pd.read_csv(m5_file)
    # airmass cut here

    idx = m5['airmass'] <= airmass_max
    m5 = m5[idx]

    m5_med = m5.groupby(['note', 'season', 'filter'])[
        'fiveSigmaDepth', 'airmass'].median().reset_index()
    idx = m5_med['note'] == field
    idxb = m5_med['season'].isin(seasonb)
    # idx &= m5_med['season'] == season
    m5_field = m5_med[idx & idxb]

    print(m5_field)

    cmd = 'python run_scripts/sim_to_fit/run_fake.py'
    cmd += ' --SN_NSNabsolute=1 --MultiprocessingSimu_nproc=8'
    cmd += ' --seasons={}'.format(seasons)

    for b in 'ugrizy':
        io = m5_field['filter'] == b
        sel = m5_field[io]
        cmd += ' --m5_{}={}'.format(b, sel['fiveSigmaDepth'].values[0])


if run_mode == 'slow':
    cmd = 'python run_scripts/sim_to_fit/run_fake.py'
    cmd += ' --SN_NSNabsolute=1 --obsFromSimu=1 --SN_daymax_type=uniform'
    cmd += ' --SN_daymax_step 3'
    cmd += ' --MultiprocessingFit_nproc=8 --MultiprocessingSimu_nproc=8'
    cmd += ' --obsCondFile={}'.format(m5_file)
    cmd += ' --seasons={}'.format(seasons)
    cmd += ' --field={}'.format(field)

print(cmd)
os.system(cmd)
