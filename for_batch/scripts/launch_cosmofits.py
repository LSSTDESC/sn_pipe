import os
from optparse import OptionParser



parser = OptionParser()
parser.add_option("--Ny", type=int, default=20,help="y-band max visits at z=0.9 [%default]")
parser.add_option("--fit_parameters", type=str, default='Om,w0',
                  help="parameters to fit [%default]")

opts, args = parser.parse_args()

Ny = opts.Ny
fit_parameters = opts.fit_parameters



configs = ['config_cosmoSN_universal_10.csv',
           'config_cosmoSN_deep_rolling_0.90_0.90_2_2.csv',
           'config_cosmoSN_deep_rolling_0.80_0.80_2_2.csv',
           'config_cosmoSN_deep_rolling_2_2_mini.csv']
           #'config_cosmoSN_deep_rolling_0.80_2.csv']
           #'config_cosmoSN_deep_rolling_0.90_0.90_3_3.csv',
           #'config_cosmoSN_deep_rolling_0.80_0.80_3_3.csv']

outDir = '/sps/lsst/users/gris/fake/Fit_bias'

scr = 'python for_batch/scripts/loop_cosmo_scen.py --fileName'

for conf in configs:
    cmd = '{} {}'.format(scr,conf)
    cmd += ' --outDir {}'.format(outDir)
    cmd += ' --Ny {}'.format(Ny)
    cmd += ' --fit_parameters {}'.format(fit_parameters)
    print(cmd)
    os.system(cmd)
