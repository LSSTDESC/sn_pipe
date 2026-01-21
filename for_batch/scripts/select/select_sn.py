#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 16:13:41 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import os
from sn_tools.sn_io import checkDir

parser = OptionParser(description='script to select SNe Ia')

parser.add_option("--dataDir", type="str",
                  default='/sps/lsst/groups/cadence/LSST_SN_PhG/prod_simu/Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass',
                  help="data dir [%default]")
parser.add_option("--listFields", type="str",
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help="list of fields to process [%default]")
parser.add_option("--fieldType", type="str",
                  default='DDF',
                  help="Type of fields to process [%default]")
parser.add_option("--dbList", type="str",
                  default='list_OS_new.csv',
                  help="List of OS to process [%default]")
parser.add_option("--timescale", type="str",
                  default='year',
                  help="timescale for output files [%default]")
parser.add_option("--selconfig", type="str",
                  default='G10_JLA',
                  help="selection criteria [%default]")
parser.add_option("--outDir_pre", type="str",
                  default='/sps/lsst/users/gris/Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass',
                  help="main output directory [%default]")
parser.add_option("--scriptName", type="str",
                  default='select_dd.sh',
                  help="output sh script [%default]")
parser.add_option("--runIt", type=int,
                  default=1,
                  help="to run the sh script [%default]")
parser.add_option("--outDir_script", type=str,
                  default='sh_scripts_run',
                  help="sh script dir [%default]")

opts, args = parser.parse_args()

pp = vars(opts)

main_script = 'python run_scripts/sn_selection/loop_selection.py'

if pp['fieldType'] == 'DDF':
    del pp['listFields']


scriptName = pp['scriptName']
del pp['scriptName']

runIt = pp['runIt']
del pp['runIt']

cmd = main_script

for key, vals in pp.items():
    to = ' --{}={}'.format(key, vals)
    cmd += to

print(cmd)
# get current directory
# cwd = os.getcwd()

# script dir
# scriptDir = '{}/{}'.format(cwd, pp['outDir_script'])
scriptDir = pp['outDir_script']
checkDir(scriptDir)

scriptName = '{}/{}'.format(scriptDir, scriptName)

# fill the script
script = open(scriptName, "w")
# script.write(qsub + "\n")
script.write("#!/bin/env bash\n")
script.write(cmd+'\n')
script.close()

# execute
cmd_e = 'sh srun_test.sh {}'.format(scriptName)

if runIt:
    st = os.stat(scriptName)
    os.chmod(scriptName, st.st_mode | 0o111)
    os.system(cmd_e)
