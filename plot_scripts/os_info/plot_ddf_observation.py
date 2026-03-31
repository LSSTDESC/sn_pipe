#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:38:03 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from sn_tools.sn_cadence_tools import get_fields
import numpy as np
from sn_plotter_analysis import plt
from sn_analysis.sn_calc_plot import bin_it
import pandas as pd

parser = OptionParser(
    description='Script to analyse DDFs on a nightly basis from pointings')

parser.add_option("--dbDir", type="str",
                  default='../DB_Files',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='baseline_v5.0.0_10yrs',
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

print(np.unique(ddf['field']))
tta = ['COSMOS', 'XMM-LSS', 'ELAISS1', 'CDFS', 'EDFS_a','EDFS_b']
lcol = ['g', 'm', 'r', 'orange', 'b','b']
lst = ['solid', 'dotted', 'dashdot', 'dashed', 'dashdot','dashdot']
dcol = dict(zip(tta, lcol))
dlst = dict(zip(tta, lst))

dcol = dict(zip(tta, lcol))

fields = np.unique(ddf['field'])
fig, ax = plt.subplots(figsize=(12,8))
fig.suptitle(dbName)
for field in fields:
    idx = ddf['field'] == field
    sel = ddf[idx]
    #ax.hist(sel['airmass'],bins=20)
    print(field,np.median(sel['airmass']))
    ro = bin_it(pd.DataFrame(sel), xvar='airmass', bins=np.arange(1.0, 2.501, 0.01),
           norm_factor=1, outvar='nobs')
    ro['nobs']/=ro['nobs'].sum()
    ro['nobs']*=100.
    ax.plot(ro['airmass'],np.cumsum(ro['nobs']),
            color=dcol[field],linestyle=dlst[field],label=field)
    print(ro)
    
ax.grid(visible=True)
ax.set_xlim([1.05,2.5])
ax.set_ylim([0.,102])
ax.set_xlabel('airmass')
ax.set_ylabel('frac of observations (<airmass) [%]')
ax.plot([1.05,2.5],[95]*2,linestyle='dotted',color='k')
ax.text(x=1.1, y=97.5, s='95%', 
                ha='center', va='center', color='k',
                backgroundcolor='white',fontsize=12)
ax.legend()
plt.show()
    
    
    
