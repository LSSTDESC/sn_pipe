from optparse import OptionParser
import pandas as pd
import os

def batch(dbName,dbDir,dbExtens,fieldName,outDir,nproc=8):

    # get config for the run
    config = get_config()

    dirScript, name_id, log, cwd = prepareOut(dbName,fieldName)
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

    for key, val in config.items():
        cmd_=cmd(dbName,dbDir,dbExtens,fieldName,config,key,outDir,nproc)
        script.write(cmd_+" \n")

    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)

def prepareOut(dbName,fieldName):
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

    id = '{}_{}'.format(dbName, fieldName)

    name_id = 'simulation_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'

    return dirScript, name_id, log, cwd


def get_config():    
    config = {}

    config['faintSN'] = {}
    config['allSN'] = {}
    for vv in ['x1','color','z']:
        config['allSN'][vv] = {}
        config['allSN'][vv]['type'] = 'random'
        config['allSN'][vv]['min'] = 0.01
        config['allSN'][vv]['max'] = 1.2

    for vv in ['x1','color','z']:
        config['faintSN'][vv] = {}
        if vv != 'z':
            config['faintSN'][vv]['type'] = 'unique'
            if vv == 'x1':
                config['faintSN'][vv]['min'] = -2.0
            if vv == 'color':
                config['faintSN'][vv]['min'] = 0.2    
            
        else:
            config['faintSN'][vv]['type'] = 'random'
            config['faintSN'][vv]['min'] = 0.01
            config['faintSN'][vv]['max'] = 0.8

    return config

def cmd(dbName,dbDir,dbExtens,fieldName,configa,key,outDir,nproc):

    config = configa[key]
    cmd = 'python run_scripts/simulation/run_simulation.py'
    cmd += ' --dbName {}'.format(dbName)
    cmd += '  --dbExtens {}'.format(dbExtens) 
    cmd += ' --dbDir {}'.format(dbDir)
    cmd += ' --Observations_fieldtype DD' 
    cmd += ' --nclusters 6 --nproc {}'.format(int(nproc))
    for vv in ['x1','color','z']:
        cmd += ' --SN_{}_type {}' .format(vv,config[vv]['type'])
    for vv in ['x1','color']:
            if config[vv]['type'] == 'unique':
                cmd += ' --SN_{}_min {}'.format(vv,config[vv]['min'])
     
    cmd += ' --SN_z_min {} --SN_z_max {}'.format(config['z']['min'], config['z']['max'])
    cmd += ' --pixelmap_dir \'\'' 
    cmd += ' --SN_NSNfactor 10'
    cmd += ' --Observations_fieldname {}'.format(fieldName)
    cmd += ' --ProductionID DD_{}_{}_error_model_{}'.format(dbName,fieldName,key)
    cmd += ' --Simulator_errorModel 1'
    cmd += ' --Output_directory {}/{}'.format(outDir,dbName)
    print(cmd)
    return cmd

parser = OptionParser()

parser.add_option("--dbName", type="str", default='descddf_v1.4_10yrs',help="dbName to process  [%default]")
parser.add_option("--dbDir", type="str", default='/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.4/npy',help="db dir [%default]")
parser.add_option("--dbExtens", type="str", default='npy',help="db extension [%default]")
parser.add_option("--fieldName", type="str", default='COSMOS',help="DD field to process [%default]")
parser.add_option("--outDir", type="str", default='/sps/lsst/users/gris/DD/Simu',help="output directory [%default]")

opts, args = parser.parse_args()

print('Start processing...')

batch(opts.dbName,opts.dbDir,opts.dbExtens,opts.fieldName,opts.outDir)
