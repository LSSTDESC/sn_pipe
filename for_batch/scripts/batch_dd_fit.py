from optparse import OptionParser
import pandas as pd
import os
import glob


def process(dbName,fieldName,prodid, simuDir, outDir,num,nproc=8,mode='batch',snrmin=5.):

    if mode == 'batch':
        batch(dbName,fieldName,prodid, simuDir, outDir,num,nproc,snrmin)
    else:
        cmd_ = cmd(dbName,prodid,simuDir,outDir,nproc,snrmin)
        os.system(cmd_)

def batch(dbName,fieldName,prodid, simuDir, outDir,num,nproc=8,snrmin=5.):

    dirScript, name_id, log, cwd = prepareOut(dbName,fieldName,num)
    # qsub command                                                                             
    qsub = 'qsub -P P_lsst -l sps=1,ct=3:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(log, nproc)

    scriptName = dirScript+'/'+name_id+'.sh'

    # fill the script                                                                          
    script = open(scriptName, "w")
    script.write(qsub + "\n")
    script.write("#!/bin/env bash\n")
    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh Linux\n")
    script.write("echo 'sourcing done' \n")

   
    cmd_=cmd(dbName,prodid, simuDir, outDir,nproc,snrmin)
    script.write(cmd_+" \n")

    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)
    

def prepareOut(dbName,fieldName,num):
    """                                                                                        
    Function to prepare for the batch                                                            
    
    directories for scripts and log files are defined here.                                    
                                                                                                   
    """

    cwd = os.getcwd()
    dirScript = cwd + "/scripts"

    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)

    id = '{}_{}_{}'.format(dbName, fieldName,num)

    name_id = 'fit_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'

    return dirScript, name_id, log, cwd


def cmd(dbName,prodid,simuDir,outDir,nproc,snrmin):
    cmd = 'python run_scripts/fit_sn/run_sn_fit.py'
    #cmd += ' --ProductionID {}_{}_allSN_{}_sn_cosmo'.format(dbName,fieldName,num) 
    #cmd += ' --Simulations_prodid {}_{}_allSN_{}'.format(dbName,fieldName,num)
    cmd += ' --ProductionID {}_sn_cosmo'.format(prodid) 
    cmd += ' --Simulations_prodid {}'.format(prodid)
    cmd += ' --Simulations_dirname {}'.format(simuDir)
    cmd += ' --LCSelection_snrmin {}'.format(snrmin) 
    cmd += ' --LCSelection_nbands 3' 
    cmd += ' --Output_directory {}/{}'.format(outDir,dbName) 
    cmd += ' --Multiprocessing_nproc {}'.format(nproc)
    return cmd

parser = OptionParser()

parser.add_option("--dbName", type="str", default='descddf_v1.4_10yrs',help="dbName to process  [%default]")
parser.add_option("--simuDir", type="str", default='/sps/lsst/users/gris/DD/Simu',help="simu dir [%default]")
parser.add_option("--fieldName", type="str", default='COSMOS',help="DD field to process [%default]")
parser.add_option("--outDir", type="str", default='/sps/lsst/users/gris/DD/Fit',help="output directory [%default]")
parser.add_option("--mode", type="str", default='batch',help="run mode batch/interactive[%default]")
parser.add_option("--snrmin", type=float, default=1.,help="min snr for LC points fit[%default]")
parser.add_option("--nproc", type=int, default=8,help="number of proc to use[%default]")

opts, args = parser.parse_args()

print('Start processing...')


#get the simufile here

simuFiles = glob.glob('{}/{}/Simu*{}*.hdf5'.format(opts.simuDir,opts.dbName,opts.fieldName))

print('hh',simuFiles,len(simuFiles))

simuDir='{}/{}'.format(opts.simuDir,opts.dbName)
for i,fi in enumerate(simuFiles):
    prodid = fi.split('/')[-1].split('.hdf5')[0].split('Simu_')[-1]
    print(prodid)
    process(opts.dbName,opts.fieldName,prodid, simuDir, opts.outDir,i,opts.nproc,opts.mode,opts.snrmin)


"""
dbName = 'descddf_v1.4_10yrs'
fieldName = 'COSMOS'


for num in range(8):
    cmd = 'python run_scripts/fit_sn/run_sn_fit.py'
    cmd += ' --ProductionID {}_{}_allSN_{}_sn_cosmo'.format(dbName,fieldName,num) 
    cmd += ' --Simulations_prodid {}_{}_allSN_{}'.format(dbName,fieldName,num)
    cmd += ' --Simulations_dirname /sps/lsst/users/gris/DD/Simu/{}'.format(dbName)
    cmd += ' --LCSelection_snrmin 5.' 
    cmd += ' --LCSelection_nbands 3' 
    cmd += ' --Output_directory /sps/lsst/users/gris/DD/Fit' 
    cmd += ' --Multiprocessing_nproc 8'
    print(cmd)
    os.system(cmd)
    
"""
