#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:49:55 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sn_tools.sn_cadence_tools import get_fields
from sn_tools.sn_obs import season


def nvisits(grp):
    
    ddict = {}
    ddict['nvisits'] = [len(grp)]
    
    for b in 'ugrizy':
        idx = grp['filter'] == b
        ddict['nvisits_{}'.format(b)] = [len(grp[idx])]
        
    res = pd.DataFrame.from_dict(ddict)
    
    return res
        
def get_seas(grp):

    res = season(grp.to_records(index=False),mjdCol='mjd')

    return pd.DataFrame.from_records(res)   

parser = OptionParser(
    description='Script to analyse DDF visits on a nightly basis from pointings')

parser.add_option("--dbDir", type="str",
                  default='../DB_Files',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='baseline_v5.0.0_10yrs',
                  help="OS name [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName


# load the data

fName = '{}/{}.npy'.format(dbDir, dbName)
obs = np.load(fName)

# grab the DDFs
ddf = get_fields(obs, lookuptable='input/simulation/lookup_ddf.csv')

ddf = pd.DataFrame.from_records(ddf)

ddf = ddf.groupby(['field']).apply(lambda x: get_seas(x),
                                   include_groups=False).reset_index()

ddfb = ddf.groupby(['field','night','season']).apply(lambda x:nvisits(x),
                                            include_groups=False).reset_index()
print(ddfb)

idx = ddfb['field'] == 'XMM-LSS'
idx &= ddfb['season'] == 5

sel = ddfb[idx]

season_length = sel['night'].max()-sel['night'].min()

fig, ax = plt.subplots()


ax.plot(sel['night'],sel['nvisits_i'],'ko')

idxb = sel['nvisits_i'] > 25

selb = sel[idxb]


season_length_ud = selb['night'].max()-selb['night'].min()

vv = 'nvisits_i'
print(season_length,season_length_ud,
      selb[vv].median(),selb[vv].mean(),selb[vv].std(),selb[vv].sum())
plt.show()
