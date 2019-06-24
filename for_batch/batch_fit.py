import os
import numpy as np
import glob

def batch(scriptref,inputDir,dbName, season,nproc):

    files = glob.glob('{}/Simu_{}_seas_{}_*'.format(inputDir,dbName,season))

    cwd = os.getcwd()
    dirScript= cwd + "/scripts"

    if not os.path.isdir(dirScript) :
        os.makedirs(dirScript)
    
    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog) :
        os.makedirs(dirLog)    
    
    
    id='{}_seas_{}'.format(dbName,season)
    name_id='fit_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'


    qsub = 'qsub -P P_lsst -l sps=1,ct=05:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(log,nproc)
    #qsub = "qsub -P P_lsst -l sps=1,ct=05:00:00,h_vmem=16G -j y -o "+ log + " <<EOF"
    scriptName = dirScript+'/'+name_id+'.sh'

    script = open(scriptName,"w")
    script.write(qsub + "\n")
    script.write("#!/usr/local/bin/bash\n")
    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh CCIN2P3\n")
    script.write("echo 'sourcing done' \n")
    cmd = 'export PYTHONPATH=sn_fit_LC:$PYTHONPATH'
    script.write(cmd+" \n")
    cmd='python {} --dbDir {} --dbName {} --season {} --nproc {}'.format(scriptref,inputDir,dbName,season,nproc)
    script.write(cmd+" \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)


inputDir = '/sps/lsst/users/gris/Output_Simu_pipeline_0'
#inputDir = '/sps/lsst/users/gris/Output_Simu_pipeline_diffflux'
files = glob.glob('{}/Simu*'.format(inputDir))

dbNames = ['alt_sched','alt_sched_rolling','kraken_2026']

for dbName in dbNames:
    #name = fi.split('/')[-1].split('.')[0]
    #prodid = '_'.join(name.split('_')[1:])
    for season in range(1,11):
        batch('run_scripts/run_sn_fit.py',inputDir,dbName,season,8)

