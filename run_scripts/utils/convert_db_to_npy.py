#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 09:53:18 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import sqlite3
import numpy as np
from optparse import OptionParser
from sn_tools.sn_obs import getObservations


parser = OptionParser(description='Script to convert db file to npy')

parser.add_option('--inputDir', type=str,
                  default='../DB_Files',
                  help='input dir[%default]')
parser.add_option('--outputDir', type=str,
                  default='../DB_Files',
                  help='outputdir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v3.0_10yrs',
                  help='DB to convert [%default]')
parser.add_option('--tableName', type=str,
                  default='observations',
                  help='Table to convert [%default]')
parser.add_option('--split_in_RA', type=int,
                  default=0,
                  help='To split obs in RA slices [%default]')


opts, args = parser.parse_args()

inputDir = opts.inputDir
outputDir = opts.outputDir
dbName = opts.dbName
tableName = opts.tableName
split_in_RA = opts.split_in_RA

# open sqlite connexion
# cnx = sqlite3.connect('{}/{}.db'.format(inputDir, dbName))

# load in df
"""
df = pd.read_sql_query("SELECT * FROM {}".format(tableName), cnx)
df['note'] = df['note'].astype('|S')
print(df.columns)
print(df.dtypes)
# dump in npy
np.save('{}/{}.npy'.format(outputDir, dbName), df.to_records(index=False))
"""

obs = getObservations(inputDir, dbName, 'db')

print(obs.dtype)

if not split_in_RA:
    np.save('{}/{}.npy'.format(outputDir, dbName), obs)
else:
    RA_slice = 10.
    deltaRA = 10.
    RAs = np.arange(0., 360.+RA_slice, RA_slice)

    for RA in RAs[:-1]:
        RAmin = np.round(RA, 1)
        RAmax = RAmin+deltaRA
        RAmax = np.round(RAmax, 1)

        RRAmin = RAmin-deltaRA
        RRAmax = RAmax+deltaRA

        if RAmin < 1.:
            RRAmin = 360.-deltaRA
        if RAmax > 359.:
            RRAmax = deltaRA

        if RAmin > 1. and RAmax < 359.:
            idx = obs['RA'] >= RRAmin
            idx &= obs['RA'] < RRAmax
            sel_obs = np.copy(obs[idx])
        else:
            idxa = obs['RA'] >= RRAmin
            idxb = obs['RA'] < RRAmax
            sel_obs = np.concatenate((obs[idxa], obs[idxb]))

        nname = '{}/{}_{}_{}.npy'.format(outputDir, dbName, RAmin, RAmax)
        np.save(nname, sel_obs)
