#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 14:55:59 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import numpy as np
from sn_tools.sn_batchutils import BatchIt

parser = OptionParser(
    description='Script to launch a set of batches for WFD pixels')

parser.add_option('--dbList', type=str,
                  default='WFD_fbs_4.3.1.csv',
                  help='list of DBs to process [%default]')
parser.add_option('--outDir', type=str,
                  default='/sps/lsst/users/gris/wfd_pixels',
                  help='dir where to save data [%default]')
parser.add_option('--proctime', type=str,
                  default='05:00:00',
                  help='max processing time [%default]')
parser.add_option('--procmem', type=str,
                  default='20G',
                  help='mem for processing')

opts, args = parser.parse_args()

dbList = opts.dbList
outDir = opts.outDir
proctime = opts.proctime
procmem = opts.procmem

# params
nside = 64
fieldType = 'WFD'
scriptref = 'python run_scripts/os_info/run_obs_strat_pixels.py'


# load DB to process

dbs = pd.read_csv(dbList, comment='#')

# loop on dbs

RAmin = 0.
RAmax = 360.
deltaRA = 36.

RAs = np.arange(RAmin, RAmax, deltaRA)

for i, row in dbs.iterrows():
    procDict = {}
    for vv in ['dbDir', 'dbName', 'dbExtens']:
        procDict[vv] = row[vv]
    procDict['nside'] = nside
    procDict['fieldType'] = fieldType
    procDict['outDir'] = '{}/{}'.format(outDir, row['dbName'])

    for j in range(len(RAs)-1):
        RA_min = RAs[j]
        RA_max = RA_min+deltaRA

        RA_min = np.round(RA_min, 2)
        RA_max = np.round(RA_max, 2)
        procName = 'WFDpix_{}_{}_{}'.format(row['dbName'], RA_min, RA_max)

        mybatch = BatchIt(processName=procName, time=proctime, mem=procmem)

        procDict['RA_min'] = RA_min
        procDict['RA_max'] = RA_min
        procDict['prodID'] = '{}.hdf5'.format(procName)

        mybatch.add_batch(scriptref, procDict)

        # go for batch
        mybatch.go_batch()
