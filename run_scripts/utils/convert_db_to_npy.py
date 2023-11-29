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

opts, args = parser.parse_args()

inputDir = opts.inputDir
outputDir = opts.outputDir
dbName = opts.dbName
tableName = opts.tableName

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
np.save('{}/{}.npy'.format(outputDir, dbName), obs)
