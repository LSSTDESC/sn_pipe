#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 12:50:11 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_tools.sn_cadence_tools import get_fields
import numpy as np
from sn_plotter_metrics import plt

parser = OptionParser(
    description='Script to analyse DDFs on a nightly basis from pointings')

parser.add_option("--dbDir", type="str",
                  default='../DB_Files',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='ddf_dither_0.8_v5.0.0_10yrs',
                  help="OS name [%default]")

opts, args = parser.parse_args()

# Load parameters
dbDir = opts.dbDir
dbName = opts.dbName

# load the data

fName = '{}/{}.npy'.format(dbDir, dbName)
obs = np.load(fName)

# grab the DDFs
ddf = get_fields(obs, lookuptable='input/simulation/lookup_ddf.csv')

print(type(ddf))

field = 'COSMOS'
idx = ddf['field'] == field

sel = ddf[idx]

night_min = np.min(sel['night'])

idx = sel['night'] <= night_min+10

sel = sel[idx]

print(sel.dtype)
fig, ax = plt.subplots(nrows=2, figsize=(12, 8))
fig.subplots_adjust(hspace=0)
ttit = dbName.split('_10yrs')[0]
title = '{} \n {}'.format(ttit, field)
fig.suptitle(title)
vara = 'RA'
varb = 'Dec'
vara_u = 'RA [deg]'
varb_u = 'Dec [deg]'

ax[0].plot(sel['night'], sel[vara], color='k',
           marker='.', markersize=12, linestyle='None')
ax[1].plot(sel['night'], sel[varb], color='k',
           marker='.', markersize=12, linestyle='None')

ax[1].set_xlabel(r'night')
ax[0].set_ylabel(r'{}'.format(vara_u))
ax[1].set_ylabel(r'{}'.format(varb_u))

for i in range(2):
    ax[i].grid(visible=True)
plt.show()
