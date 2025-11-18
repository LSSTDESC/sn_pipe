#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:41:11 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from sn_tools.sn_batchutils import BatchIt

sigma_pwv = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
sigma_ozone = [1., 5., 15., 20., 25.]
sigma_aerosol = [1.e-3, 5.e-3, 0.01, 0.015, 0.02, 0.025]


dfa = pd.DataFrame(sigma_pwv, columns=['sigma_pwv'])
dfb = pd.DataFrame(sigma_ozone, columns=['sigma_ozone'])
dfc = pd.DataFrame(sigma_aerosol, columns=['sigma_aerosol'])

df = dfa.merge(dfb, how='cross')

df = df.merge(dfc, how='cross')

df['num_combi'] = df.index+1
print(df)
ntrial = 20

script = 'run_scripts/telescope/zero_points_atmos.py'
outDir = '/sps/lsst/users/gris/zp_atmos'
for i, row in df.iterrows():
    num_combi = int(row['num_combi'])
    processName = 'zp_atmos_{}'.format(num_combi)
    mybatch = BatchIt(processName=processName)

    dd = {}
    for tt in ['pwv', 'ozone', 'aerosol']:
        dd['sigma_{}_min'.format(tt)] = row['sigma_{}'.format(tt)]

    dd['ntrial'] = ntrial
    dd['outDir'] = outDir
    dd['outName'] = 'zp_atmos_config{}.hdf5'.format(num_combi)

    mybatch.add_batch(script, dd)

    mybatch.go_batch()
