#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:36:13 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from optparse import OptionParser
import os


def go_batch(script, params):

    cmd = 'python {}'.format(script)
    for key, vals in params.items():
        cmd += ' --{} {}'.format(key, vals)

    print(cmd)
    os.system(cmd)


parser = OptionParser()

parser.add_option("--dbList", type="str", default='DD_fbs_2.99.csv',
                  help="db list to process[%default]")


opts, args = parser.parse_args()

# load csv file

df = pd.read_csv(opts.dbList, comment='#')

print(df)

procDict = {}
script = 'for_batch/scripts/sn_prod/sn_prod.py'
for i, row in df.iterrows():
    procDict['dbDir'] = row['dbDir']
    procDict['dbExtens'] = row['dbExtens']
    procDict['dbName'] = row['dbName']
    procDict['fieldType'] = row['fieldType']
    procDict['nside'] = row['nside']
    go_batch(script, procDict)
