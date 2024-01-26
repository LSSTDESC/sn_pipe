#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 08:52:33 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
from optparse import OptionParser
import glob
import matplotlib.pyplot as plt
import numpy as np

parser = OptionParser(
    description='Script to plot a Hubble Diagram from a simulated survey')

parser.add_option('--dbDir', type=str, default='Test_survey',
                  help='survey location dir[%default]')
parser.add_option('--dbName_DD', type=str,
                  default='DDF_DESC_0.80_SN_0.07',
                  help='OS DDF name[%default]')
parser.add_option('--dbName_WFD', type=str,
                  default='baseline_v3.0_10yrs',
                  help='OS WFD name[%default]')
parser.add_option('--year_min', type=int,
                  default=1,
                  help='min survey year[%default]')
parser.add_option('--year_max', type=int,
                  default=10,
                  help='max survey year[%default]')
parser.add_option('--timescale', type=str,
                  default='season',
                  help='timescale for plot - year or season [%default]')
"""
parser.add_option('--UDFs', type=str,
                  default='COSMOS,XMM-LSS',
                  help='UD fields [%default]')
parser.add_option('--DFs', type=str,
                  default='CDFS,EDFS,ELAISS1',
                  help='Deep fields [%default]')
"""

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName_DD = opts.dbName_DD
dbName_WFD = opts.dbName_WFD
year_min = opts.year_min
year_max = opts.year_max
timescale = opts.timescale


fullName = '{}/survey_sn_{}_{}_{}_{}*.hdf5'.format(
    dbDir, dbName_DD, dbName_WFD, year_min, year_max)

print('search_path', fullName)
tt = glob.glob(fullName)

df = pd.DataFrame()

for vv in tt:
    dfa = pd.read_hdf(vv)
    df = pd.concat((df, dfa))

# get random realization numbers
nrands = df['survey_real'].unique()

# choose one realization
io = np.random.choice(nrands, 1)[0]
print(nrands, io)

# select this survey
idx = df['survey_real'] == io
survey = df[idx]

# now plot it
seasons = survey[timescale].unique()

"""
for seas in seasons:
    idb = survey[timescale] == seas
    survey_season = survey[idb]
    fig, ax = plt.subplots()
    fig.suptitle('{} {} - Nsn={}'.format(timescale, seas, len(survey_season)))
    ax.errorbar(survey_season['z_fit'],
                survey_season['mu'], yerr=survey_season['sigma_mu'], linestyle='None', color='k')


fig, ax = plt.subplots()
fig.suptitle('{} {}-{} - Nsn={}'.format(timescale,
             year_min, year_max, len(survey)))
ax.errorbar(survey['z_fit'],
            survey['mu_SN'],
            yerr=survey['sigma_mu'],
            linestyle='None', color='k')
ax.plot(survey['z_fit'], survey['mu_fit'],
        color='r', marker='.', linestyle='None')

figb, axb = plt.subplots()
axb.hist(survey['delta_mu'], histtype='step', bins=80)
"""
survey['SNR_mu'] = survey['mu_SN']/survey['sigma_mu']
fig, ax = plt.subplots()
idx = survey['zType'] == 'spectroz'
ax.hist(1./survey[idx]['SNR_mu'], histtype='step', bins=80)
ax.hist(1./survey[~idx]['SNR_mu'], histtype='step', bins=80)

plt.show()
