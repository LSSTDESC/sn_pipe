import os

configs = ['config_cosmoSN_universal_10.csv',
           'config_cosmoSN_deep_rolling_0.90_0.90_2_2.csv',
           'config_cosmoSN_deep_rolling_0.80_0.80_2_2.csv',
           'config_cosmoSN_deep_rolling_0.90_2.csv',
           'config_cosmoSN_deep_rolling_0.80_2.csv',
           'config_cosmoSN_deep_rolling_0.90_0.90_3_3.csv',
           'config_cosmoSN_deep_rolling_0.80_0.80_3_3.csv']

outDir = '/sps/lsst/users/gris/fake/Fit_bias'

scr = 'python for_batch/scripts/loop_cosmo_scen.py --fileName'

for conf in configs:
    cmd = '{} {}'.format(scr,conf)
    cmd += ' --outDir {}'.format(outDir)
    print(cmd)
    os.system(cmd)
