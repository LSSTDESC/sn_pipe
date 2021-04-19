import os
import numpy as np
import glob
from optparse import OptionParser
import pandas as pd

def launch_batches(fis,dirSimu,dirOut,n_per_job,mbcov=0):
        
    
    nz = len(fis)
    nbatch = int(nz/n_per_job)
    t = np.linspace(0, nz,nbatch+1, dtype='int')

    for j in range(nbatch):
        
        fib = fis[t[j]:t[j+1]]
        prodids = []
        for fi in fib:
            prodid=fi.split('/')[-1].split('.hdf5')[0]
            prodid='_'.join(prodid.split('_')[1:])
            print(prodid)
            prodids.append(prodid)
        batch('run_scripts/fit_sn/run_sn_fit.py', dirSimu,prodids,dirOut, 8,mbcov)  


def batch(scriptref, inputDir, prodids, dirOut, nproc,mbcov=0):


    cwd = os.getcwd()
    dirScript = cwd + "/scripts"

    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)

    name_id = 'fit_{}_{}'.format(prodids[0],mbcov)
    log = dirLog + '/'+name_id+'.log'

    qsub = 'qsub -P P_lsst -l sps=1,ct=20:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
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
    #need this to limit the number of multithread
    script.write(" export MKL_NUM_THREADS=1 \n")
    script.write(" export NUMEXPR_NUM_THREADS=1 \n")
    script.write(" export OMP_NUM_THREADS=1 \n")
    script.write(" export OPENBLAS_NUM_THREADS=1 \n")

    for prodid in prodids:
        cmd = 'python {}'.format(scriptref)
        cmd += ' --Simulations_dirname {}'.format(inputDir) 
        cmd += ' --Simulations_prodid {}'.format(prodid) 
        cmd += ' --OutputFit_directory {}'.format(dirOut)
        cmd += ' --MultiprocessingFit_nproc {}'.format(nproc)
        cmd += ' --ProductionID {}_{}'.format(prodid,mbcov)
        cmd += ' --mbcov_estimate {}'.format(mbcov)
        cmd += ' --LCSelection_naft 5'
        cmd += ' --LCSelection_nbands 0'
        cmd += ' --LCSelection_phasemin -5.0'
        cmd += ' --LCSelection_phasemax 20.0'
        cmd += ' --LCSelection_nphasemin 1'
        cmd += ' --LCSelection_nphasemax 1'
        cmd += ' --LCSelection_errmodrel 0.1'

        script.write(cmd+" \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)

parser = OptionParser()

parser.add_option("--dbList", type="str", default='WFD.csv',
                  help="dbList to process  [%default]")

parser.add_option("--dirSimu", type="str", default='/sps/lsst/users/gris/Simulation_fbs15_circular_dust',
                  help="db dir [%default]")

parser.add_option("--dirOut", type="str", default='/sps/lsst/users/gris/Fit_fbs15_circular_dust',
                  help="output directory [%default]")

parser.add_option("--mbcov_estimate", type=int, default=0,
                  help="to activate mbcov estimate[%default]")

parser.add_option("--n_per_job", type=int, default=2,
                  help="nfiles per job [%default]")


opts, args = parser.parse_args()

print('Start processing...')

toproc=pd.read_csv(opts.dbList,comment='#')
n_per_job = opts.n_per_job
mbcov = opts.mbcov_estimate


for index, row in toproc.iterrows():
    #name = fi.split('/')[-1].split('.')[0]
    #prodid = '_'.join(name.split('_')[1:])
    # get the number of files
    dbName = row['dbName']
    search_path = '{}/{}/Simu*SN_Ia*{}*.hdf5'.format(opts.dirSimu,dbName,dbName)
    print('search path',search_path)
    fis = glob.glob(search_path)
    if len(fis)>0:
        print(dbName,len(fis))
        dirSimu = '{}/{}'.format(opts.dirSimu,dbName)
        dirOut = '{}/{}'.format(opts.dirOut,dbName)
        launch_batches(fis,dirSimu,dirOut,n_per_job,mbcov=mbcov)


           
                   

    """
    for season in range(1, 11):
        batch('run_scripts/run_sn_fit.py', inputDir, dbName, season, 8)
    """
