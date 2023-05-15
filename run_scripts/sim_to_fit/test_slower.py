#!/bin/bash
#!/bin/bash
import os
import pandas as pd
from optparse import OptionParser


parser = OptionParser()

parser.add_option("--m5_file", type=str,
                  default='input/m5_OS/m5_field_night_baseline_v3.0_10yrs.csv',
                  help="file to process [%default]")

opts, args = parser.parse_args()


m5_file = opts.m5_file

cmd = 'python run_scripts/sim_to_fit/run_fake.py'
cmd += ' --SN_NSNabsolute=1 --obsFromSimu=1 --SN_daymax_type=uniform'
cmd += ' --MultiprocessingFit_nproc=8 --MultiprocessingSimu_nproc=8'
cmd += ' --obsCondFile={}'.format(m5_file)

print(cmd)
os.system(cmd)
