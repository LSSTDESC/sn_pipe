#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 14:34:33 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import os
from sn_tools.sn_io import checkDir
from sn_tools.sn_batchutils import open_script,add_script

parser = OptionParser(
    description='Script to estimate the total number of simulated SNe Ia on a set of files')
parser.add_option('--dbDir', type=str,
                  default='/sps/lsst/groups/cadence/LSST_PhG/prod_simu/Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass',
                  help='OS location dir [%default]')
parser.add_option('--dbList', type=str,
                  default='dbList.csv',
                  help='OS to process [%default]')
parser.add_option('--runType', type=str,
                  default='WFD_spectroz_nosat',
                  help='type of run [%default]')
parser.add_option('--fieldType', type=str,
                  default='WFD',
                  help='type of field (DDF/WFD) [%default]')
parser.add_option('--outDir', type=str,
                  default='/sps/lsst/users/gris/nsn_simu_prod',
                  help='outDir [%default]')
parser.add_option('--outName_pre', type=str,
                  default='nsn_simu',
                  help='output file name (csv) [%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='number of procs for multiprocessing [%default]')
parser.add_option('--shDir', type=str,
                  default='sh_scripts',
                  help='dir for sh scripts [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
runType = opts.runType
outDir = opts.outDir
outName_pre = opts.outName_pre
nproc = opts.nproc
shDir = opts.shDir
fieldType = opts.fieldType

#check shdir
checkDir(shDir)

#load the DBs to process

df_db = pd.read_csv(dbList,comment='#')

print(df_db)
#create the scripts
scriptref = 'run_scripts/sn_analysis/nsn_simu.py'
procDict = {}
procDict['dbDir'] = opts.dbDir
procDict['runType'] = opts.runType
procDict['outDir'] = outDir
procDict['nproc'] = nproc

procName = 'nsn_simu_{}'.format(fieldType)
scriptName = '{}/{}.sh'.format(shDir, procName)
script = open_script(scriptName)

for i, row in df_db.iterrows():
    dbName = row['dbName']
    procName = 'nsn_simu_{}_{}'.format(fieldType,dbName)
    procDict['dbName'] = dbName
    procDict['outName'] = '{}_{}_{}.csv'.format(outName_pre,fieldType,dbName)
    add_script(script,scriptref,procDict)

script.close()
st = os.stat(scriptName)
os.chmod(scriptName, st.st_mode | 0o111)
cmd_e = 'sh srun_test.sh {}'.format(scriptName)
os.system(cmd_e)