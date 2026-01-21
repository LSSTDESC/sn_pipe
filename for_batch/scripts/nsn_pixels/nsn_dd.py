#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 15:15:21 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
from sn_tools.sn_batchutils import BatchIt
import numpy as np
import os
from sn_tools.sn_io import checkDir


def make_script(scriptDir, scriptName, **pp):
    """
    Function to write a script to be used in interactive

    Parameters
    ----------
    scriptDir : str
        script directory.
    scriptName : str
        script name.
    **pp : dict
        script parameters.

    Returns
    -------
    None.

    """

    # get current directory
    cwd = os.getcwd()

    # script dir
    scriptDir = '{}/{}'.format(cwd, scriptDir)
    checkDir(scriptDir)

    scriptName = '{}/{}'.format(scriptDir, scriptName)

    # fill the script
    script = open(scriptName, "w")

    main_script = 'python run_scripts/sn_analysis/nsn_ddf.py'

    cmd = main_script

    for key, vals in pp.items():
        to = ' --{}={}'.format(key, vals)
        cmd += to

    script.write("#!/bin/env bash\n")
    script.write(cmd+'\n')
    script.close()


def make_batch(dbList, procDict, time='5:00:00', mem='50G', inum=0):
    """
    Function to launch batch

    Parameters
    ----------
    dbList : pandas df
        List of OS to process.
    procDict : dict
        parameter dict.
    time : str, optional
        Time for the batch. The default is '5:00:00'.
    mem : str, optional
        memory for the batch. The default is '30G'.
    inum : int, optional
        tag for the process. The default is 0.

    Returns
    -------
    None.

    """

    procName = 'nsn_dd_{}'.format(inum)

    mybatch = BatchIt(processName=procName, time=time, mem=mem)

    csvDir = mybatch.logDir
    csvName = '{}/DD_list_{}.csv'.format(csvDir, inum)

    dbList.to_csv(csvName, index=None)

    scriptref = 'run_scripts/sn_analysis/nsn_dd.py'
    # for vv in dbList:
    procDict['dbList'] = csvName
    mybatch.add_batch(scriptref, procDict)

    mybatch.go_batch()


def runBatch(procDict):
    """
    Function to run in batch

    Parameters
    ----------
    procDict : dict
        parameter dict.

    Returns
    -------
    None.

    """

    dbNames = pd.read_csv(procDict['dbList'], comment='#')

    n_split = 1
    if len(dbNames) >= 4:
        n_split = int(len(dbNames)/4)
    chunks = np.array_split(dbNames, n_split)

    print(chunks)

    for i, val in enumerate(chunks):
        print(i, val)
        make_batch(val, procDict, inum=i)


parser = OptionParser(
    description='Script to estimate nsn for DD - in batch/interactive')

parser.add_option('--dbDir', type=str,
                  default='/sps/lsst/users/gris/Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str,
                  default='list_dd.csv',
                  help='list of OS to process [%default]')
parser.add_option('--outDir', type=str,
                  default='/sps/lsst/users/gris/sn_dd_airmass',
                  help='output dir [%default]')
parser.add_option("--scriptName", type="str",
                  default='nsn_dd.sh',
                  help="output sh script [%default]")
parser.add_option("--scriptDir", type="str",
                  default='sh_scripts_run',
                  help="output sh script dir [%default]")
parser.add_option("--runIt", type=int,
                  default=1,
                  help="to run the sh script [%default]")
parser.add_option("--runMode", type=str,
                  default='interactive',
                  help="run mode (interactive/batch) [%default]")

opts, args = parser.parse_args()

procDict = vars(opts)

if procDict['runMode'] == 'batch':
    for vv in ['scriptName', 'scriptDir', 'runIt', 'runMode']:
        del procDict[vv]
    runBatch(procDict)
else:
    scriptDir = procDict['scriptDir']
    scriptName = procDict['scriptName']
    runIt = procDict['runIt']

    for vv in ['scriptName', 'scriptDir', 'runIt', 'runMode']:
        del procDict[vv]
    make_script(scriptDir, scriptName, **procDict)

    # execute
    if runIt:
        fName = '{}/{}'.format(scriptDir, scriptName)
        cmd_e = 'sh srun_test.sh {}'.format(fName)
        st = os.stat(fName)
        os.chmod(fName, st.st_mode | 0o111)
        os.system(cmd_e)


"""
# loop and batch
for i, row in dbNames.iterrows():
    print(row)
"""
