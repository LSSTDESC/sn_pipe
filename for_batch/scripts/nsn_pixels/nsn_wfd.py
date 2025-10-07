#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 15:15:21 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
from sn_tools.sn_batchutils import BatchIt

def make_batch(dbList,procDict,time='5:00:00', mem='5G',inum=0):
    
    procName = 'nsn_wfd_{}'.format(inum)
    mybatch = BatchIt(processName=procName, time=time, mem=mem)

    scriptref = 'run_scripts/sn_analysis/nsn_wfd.py'
    for vv in dbList:
        mybatch.add_batch(scriptref, procDict)

    mybatch.go_batch()

parser = OptionParser(description='Script to estimate nsn for WFD - in batch')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str,
                  default='list_wfd.csv',
                  help='list of OS to process [%default]')
parser.add_option('--outDir', type=str,
                  default='../sn_wfd',
                  help='output dir [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
outDir = opts.outDir

#load data

dbNames = pd.read_csv(dbList,comment='#')

# loop and batch
for i, row in dbNames.iterrows():
    print(row)
    
    
    
    