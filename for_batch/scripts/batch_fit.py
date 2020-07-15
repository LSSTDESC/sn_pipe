import os
import numpy as np
import glob
from optparse import OptionParser
import pandas as pd

def batch(scriptref, inputDir, prodids, outDir, nproc):


    cwd = os.getcwd()
    dirScript = cwd + "/scripts"

    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)

    name_id = 'fit_{}'.format(prodids[0])
    log = dirLog + '/'+name_id+'.log'

    qsub = 'qsub -P P_lsst -l sps=1,ct=10:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
        log, nproc)
    #qsub = "qsub -P P_lsst -l sps=1,ct=05:00:00,h_vmem=16G -j y -o "+ log + " <<EOF"
    scriptName = dirScript+'/'+name_id+'.sh'

    script = open(scriptName, "w")
    script.write(qsub + "\n")
    # script.write("#!/usr/local/bin/bash\n")
    script.write("#!/bin/env bash\n")
    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh Linux\n")
    script.write("echo 'sourcing done' \n")
    for prodid in prodids:
        cmd = 'python {} --dirFiles {} --prodid {} --outDir {} --nproc {}'.format(
            scriptref, inputDir, prodid , outDir, nproc)
        script.write(cmd+" \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)

parser = OptionParser()

parser.add_option("--dbList", type="str", default='WFD.csv',
                  help="dbList to process  [%default]")

parser.add_option("--dbDir", type="str", default='/sps/lsst/users/gris/Simulation_fbs15_circular_dust',
                  help="db dir [%default]")

parser.add_option("--outDir", type="str", default='/sps/lsst/users/gris/Fit_fbs15_circular_dust',
                  help="output directory [%default]")

opts, args = parser.parse_args()

print('Start processing...')

toproc=pd.read_csv(opts.dbList)

for index, row in toproc.iterrows():
    #name = fi.split('/')[-1].split('.')[0]
    #prodid = '_'.join(name.split('_')[1:])
    # get the number of files
    fis = glob.glob('{}/Simu*{}*.hdf5'.format(opts.dbDir,row['dbName']))
    if len(fis)>0:
        print(row['dbName'],len(fis))
        for fi in fis:
            prodid=fi.split('/')[-1].split('.hdf5')[0]
            prodid='_'.join(prodid.split('_')[1:])
            print(prodid)
            batch('run_scripts/fit_sn/run_sn_fit.py', opts.dbDir,[prodid], opts.outDir, 8)             
                   

    """
    for season in range(1, 11):
        batch('run_scripts/run_sn_fit.py', inputDir, dbName, season, 8)
    """
