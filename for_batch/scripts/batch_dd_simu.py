from optparse import OptionParser
import pandas as pd
import os
import numpy as np
import itertools

def process(dbName,dbDir,dbExtens,fieldName,outDir,pixelmap_dir,ebvofMW=-1.,nproc=8,mode='batch'):
    
    if mode == 'batch':
        batch(dbName,dbDir,dbExtens,fieldName,outDir,pixelmap_dir,ebvofMW,nproc)
    else:
         config = config_rec()
         confNames = ['faintSN','allSN']
         for cf in confNames:
             idx = config['confName'] == cf
             sel = config[idx]
             print('alors',sel)
             spl = np.array_split(sel,5)
             for ibatch,vv in enumerate(spl):
                 for val in vv:
                     cmd_=cmd(dbName,dbDir,dbExtens,fieldName,val,outDir,pixelmap_dir,0,0,nproc)
                     print(cmd_)
                     os.system(cmd_)


def batch(dbName,dbDir,dbExtens,fieldName,outDir,pixelmap_dir,ebvofMW,nproc=8):
    # get config for the run
    config = config_rec()

    
    #confNames = ['faintSN','allSN']
    #confNames = ['mediumSN']
    confNames = ['allSN']
    for cf in confNames:
        idx = config['confName'] == cf
        sel = config[idx]
        spl = np.array_split(sel,np.min([5,len(sel)]))
        for ibatch,vv in enumerate(spl):
            batch_indiv(dbName,dbDir,dbExtens,fieldName,outDir,pixelmap_dir,ebvofMW,cf,vv,ibatch,nproc)


def batch_indiv(dbName,dbDir,dbExtens,fieldName,outDir,pixelmap_dir,ebvofMW,confname,config,ibatch,nproc=8):

    dirScript, name_id, log, cwd = prepareOut(dbName,fieldName,confname,ibatch)
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

    for iconf,val in enumerate(config):
        cmd_=cmd(dbName,dbDir,dbExtens,fieldName,val,outDir,pixelmap_dir,ebvofMW,ibatch,iconf,nproc)
        script.write(cmd_+" \n")

    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)

def prepareOut(dbName,fieldName,confname,ibatch):
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

    id = '{}_{}_{}_{}'.format(dbName, fieldName,confname,ibatch)

    name_id = 'simulation_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'

    return dirScript, name_id, log, cwd

def config_rec():

    zmin = 0.0
    zmax = 1.2
    zstep = 0.05

    r = []
    
    x1_type = 'unique'
    x1_min = -2.0
    x1_max = 2.0
    color_type = 'unique'
    color_min = 0.2
    color_max = 0.4
    z_type = 'uniform'
    daymax_type = 'uniform'
    daymax_step = 2
    nsn_factor =1

    # this is for faintSN
    z_step = 0.5
    for z in np.arange(zmin,1.0,z_step):
        z = np.round(z,2)
        z_min = z
        z_max = z+np.round(z_step,2)
        r.append((x1_type,x1_min,x1_max,color_type,color_min,color_max,z_type,z_min,z_max,zstep,daymax_type, daymax_step,nsn_factor,'faintSN'))


    for z in np.arange(zmin,zmax,zstep):
        zval = np.round(z,2)
        x1_type = 'unique'
        x1_min = 0.0
        x1_max = 2.0
        color_type = 'unique'
        color_min = 0.0
        color_max = 0.4
        z_type = 'random'
        day_max = 'random'
        z_min = zval
        z_max = zval+np.round(zstep,2)
        z_step = np.round(zstep,2)
        nsn_factor = 105

        r.append((x1_type,x1_min,x1_max,color_type,color_min,color_max,z_type,z_min,z_max,z_step,daymax_type,daymax_step,nsn_factor,'mediumSN'))

        z_type = 'random'
        x1_type = 'random'
        color_type = 'random'
        daymax_type = 'random'
       
        
        r.append((x1_type,x1_min,x1_max,color_type,color_min,color_max,z_type,z_min,z_max,z_step,daymax_type,daymax_step,nsn_factor,'allSN'))

    res = np.rec.fromrecords(r, names=['x1_type','x1_min','x1_max',
                                       'color_type','color_min','color_max',
                                       'z_type','z_min','z_max','z_step',
                                       'daymax_type','daymax_step','nsn_factor',
                                       'confName'])

    return res

def get_config_deprecated():    
    config = {}

    config['faintSN'] = {}
    config['allSN'] = {}

    zmin = 0.0
    zmax =1.2
    zstep =0.05

    for z in np.arange(zmin,zmax,zstep):
        zval = np.round(z,2)
        config['allSN'][zval] = {}
        config['faintSN'][zval] = {}
                    
        for vv in ['x1','color','z']:
            config['allSN'][zval][vv] = {}
            config['allSN'][zval][vv]['type'] = 'random'
            config['allSN'][zval][vv]['min'] = zval
            config['allSN'][zval][vv]['max'] = zval+zstep


        for vv in ['x1','color','z']:
            config['faintSN'][zval][vv] = {}
    
            if vv != 'z':
                config['faintSN'][zval][vv]['type'] = 'unique'
                if vv == 'x1':
                    config['faintSN'][zval][vv]['min'] = -2.0
                if vv == 'color':
                    config['faintSN'][zval][vv]['min'] = 0.2
            
            else:
                config['faintSN'][zval][vv]['type'] = 'random'
                config['faintSN'][zval][vv]['min'] = zval
                config['faintSN'][zval][vv]['max'] = zval+zstep


    return config

def cmd(dbName,dbDir,dbExtens,fieldName,config,outDir,pixelmap_dir,ebvofMW,ibatch,iconfig,nproc):

    cmd = 'python run_scripts/simulation/run_simulation.py'
    cmd += ' --dbName {}'.format(dbName)
    cmd += '  --dbExtens {}'.format(dbExtens) 
    cmd += ' --dbDir {}'.format(dbDir)
    cmd += ' --Observations_fieldtype DD' 
    cmd += ' --nclusters 6 --nproc {}'.format(int(nproc))
    for vv in ['x1','color','z']:
        cmd += ' --SN_{}_type {}' .format(vv,config['{}_type'.format(vv)])
    for vv in ['x1','color']:
            if config['{}_type'.format(vv)] == 'unique':
                cmd += ' --SN_{}_min {}'.format(vv,config['{}_min'.format(vv)])
     
    cmd += ' --SN_z_min {} --SN_z_max {} --SN_z_step {}'.format(config['z_min'], config['z_max'],config['z_step'])
    cmd += ' --SN_daymax_type {} --SN_daymax_step {}'.format(config['daymax_type'],config['daymax_step'])
    cmd += ' --pixelmap_dir {}'.format(pixelmap_dir)
    cmd += ' --SN_NSNfactor {}'.format(config['nsn_factor'])
    cmd += ' --Observations_fieldname {}'.format(fieldName)
    cmd += ' --ProductionID DD_{}_{}_error_model_{}_{}_{}'.format(dbName,fieldName,config['confName'],ibatch,iconfig)
    cmd += ' --Simulator_errorModel 1'
    outputDir = '{}/{}'.format(outDir,dbName)
    cmd += ' --Output_directory {}'.format(outputDir)
    cmd += ' --SN_minRFphaseQual -15.'
    cmd += ' --SN_maxRFphaseQual 30.'
    cmd += ' --SN_ebvofMW {}'.format(ebvofMW)

    #create outputDir here
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)


    print(cmd)
    return cmd

parser = OptionParser()

parser.add_option('--dbName', type='str', default='descddf_v1.4_10yrs',help='dbName to process  [%default]')
parser.add_option('--dbDir', type='str', default='/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.4/npy',help='db dir [%default]')
parser.add_option('--dbExtens', type='str', default='npy',help='db extension [%default]')
parser.add_option('--fieldName', type='str', default='COSMOS',help='DD field to process [%default]')
parser.add_option('--outDir', type='str', default='/sps/lsst/users/gris/DD/Simu',help='output directory [%default]')
parser.add_option('--mode', type='str', default='batch',help='running mode batch/interactive[%default]')
parser.add_option('--pixelmap_dir', type='str', default='/sps/lsst/users/gris/ObsPixelized',help='pixelmap directory [%default]')
parser.add_option('--nproc', type=int, default=8,help='number of proc [%default]')
parser.add_option('--ebvofMW', type=float, default=-1.0,help='E(B-V) [%default]')

opts, args = parser.parse_args()

print('Start processing...')

process(opts.dbName,opts.dbDir,opts.dbExtens,opts.fieldName,opts.outDir,opts.pixelmap_dir,opts.ebvofMW,opts.nproc,opts.mode)
