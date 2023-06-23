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
parser.add_option("--outputDir", type="str", default='/sps/lsst/users/gris/Output_SN',
                  help="main output dir [%default]")
parser.add_option("--Fitter_parnames", type="str", default='t0,x1,c,x0',
                  help="parameters to fit [%default]")
parser.add_option("--reprocList", type="str", default='None.csv',
                  help="to reproc some of the db files only [%default]")
parser.add_option("--SN_sigmaInt", type=float, default=0.0,
                  help="SN intrinsic dispersion [%default]")
parser.add_option("--OutputSimu_directory", type=str, default='Output_SN',
                  help="Output directory  [%default]")


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
    procDict['OutputSimu_directory'] = opts.outputDir
    procDict['Fitter_parnames'] = opts.Fitter_parnames
    procDict['reprocList'] = opts.reprocList
    procDict['SN_sigmaInt'] = opts.SN_sigmaInt
    procDict['OutputSimu_directory'] = opts.OutputSimu_directory
    procDict['OutputFit_directory'] = opts.OutputSimu_directory
    go_batch(script, procDict)
