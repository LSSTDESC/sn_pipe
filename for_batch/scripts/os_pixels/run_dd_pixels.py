#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 14:55:59 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import numpy as np
import os
from sn_tools.sn_batchutils import BatchIt
from sn_tools.sn_io import checkDir


def open_script(fName):
    """
    Function to open a script

    Parameters
    ----------
    fName : str
        script name.

    Returns
    -------
    script : file
        script file.

    """

    # fill the script
    script = open(scriptName, "w")

    script.write("#!/bin/env bash\n")
    # script.write(cmd+'\n')

    return script


def add_script(script, main_cmd, pp):
    """
    Function to fill the script

    Parameters
    ----------
    script : file
        script to fill.
    main_cmd : str
        cmd .
    pp : dict
        parameter dict for cmd.

    Returns
    -------
    None.

    """

    cmd = 'python {}'.format(main_cmd)

    for key, vals in pp.items():
        cmd += ' --{}={}'.format(key, vals)

    script.write(cmd+'\n')


parser = OptionParser(
    description='Script to launch a set of batches for DD pixels')

parser.add_option('--dbList', type=str,
                  default='DD_fbs_4.3.1.csv',
                  help='list of DBs to process [%default]')
parser.add_option('--outDir', type=str,
                  default='/sps/lsst/users/gris/dd_pixels',
                  help='dir where to save data [%default]')
parser.add_option('--proctime', type=str,
                  default='05:00:00',
                  help='max processing time [%default]')
parser.add_option('--procmem', type=str,
                  default='20G',
                  help='mem for processing [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,XMM-LSS,CDFS,ELAISS1,EDFS_a,EDFS_b',
                  help='DDF to process [%default]')
parser.add_option('--procmode', type=str,
                  default='batch',
                  help='mode of processing: batch/interact [%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='nproc for multiprocessing [%default]')
parser.add_option('--shDir', type=str,
                  default='sh_scripts',
                  help='dir for sh scripts [%default]')

opts, args = parser.parse_args()

dbList = opts.dbList
outDir = opts.outDir
proctime = opts.proctime
procmem = opts.procmem
fields = opts.fields.split(',')
procmode = opts.procmode
nproc = opts.nproc
shDir = opts.shDir

if procmode == 'interact':
    checkDir(shDir)


# params
nside = 128
fieldType = 'DD'
scriptref = 'run_scripts/os_info/run_obs_strat_pixels.py'


# load DB to process

dbs = pd.read_csv(dbList, comment='#')

# loop on dbs

for i, row in dbs.iterrows():
    procDict = {}
    for vv in ['dbDir', 'dbName', 'dbExtens']:
        procDict[vv] = row[vv]
    procDict['nside'] = nside
    procDict['fieldType'] = fieldType
    procDict['outDir'] = '{}/{}'.format(outDir, row['dbName'])

    procName = 'DD_pixels_{}'.format(row['dbName'])

    if procmode == 'batch':
        mybatch = BatchIt(processName=procName, time=proctime, mem=procmem)

    if procmode == 'interact':
        scriptName = '{}/{}.sh'.format(shDir, procName)
        script = open_script(scriptName)

    for field in fields:

        procDict['fieldName'] = field
        prodId = '{}_{}'.format(procName, field)

        procDict['prodID'] = prodId
        procDict['nproc'] = nproc
        procDict['nproc_pixels'] = 0

        if procmode == 'batch':
            mybatch.add_batch(scriptref, procDict)

        if procmode == 'interact':
            add_script(script, scriptref, procDict)

    # go for batch
    if procmode == 'batch':
        mybatch.go_batch()
    if procmode == 'interact':
        script.close()
        st = os.stat(scriptName)
        os.chmod(scriptName, st.st_mode | 0o111)
        cmd_e = 'sh srun_test.sh {}'.format(scriptName)
        os.system(cmd_e)
