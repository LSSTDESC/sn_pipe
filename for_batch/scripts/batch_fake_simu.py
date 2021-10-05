from optparse import OptionParser
import pandas as pd
import os
import numpy as np
import itertools


def process(dbName, dbDir, dbExtens, outDir, nproc=8, batch=True, snTypes=['faintSN', 'allSN'],nabs=dict(zip(['faintSN', 'allSN'],[-1,-1])),nsnfactor=dict(zip(['faintSN', 'allSN'],[100,100])),x1sigma=0,colorsigma=0):
    # get config for the run
    config = config_rec(nabs,nsnfactor,x1sigma,colorsigma)

    #confNames = ['faintSN','allSN']
    #confNames = ['faintSN']
    confNames = snTypes
    for cf in confNames:
        idx = config['confName'] == cf
        sel = config[idx]
        spl = np.array_split(sel, np.min([1, len(sel)]))
        for ibatch, vv in enumerate(spl):
            process_indiv(dbName, dbDir, dbExtens, outDir,
                          cf, vv, ibatch, nproc, batch)


def process_indiv(dbName, dbDir, dbExtens, outDir, confname, config, ibatch, nproc=8, batch=True):

    dirScript, name_id, log, cwd = prepareOut(
        dbName, confname, ibatch)
    # qsub command
    qsub = 'qsub -P P_lsst -l sps=1,ct=3:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
        log, nproc)

    scriptName = dirScript+'/'+name_id+'.sh'

    # fill the script
    script = open(scriptName, "w")
    if batch:
        script.write(qsub + "\n")
    script.write("#!/bin/env bash\n")
    if batch:
        script.write(" cd " + cwd + "\n")
        script.write(" echo 'sourcing setups' \n")
        script.write(" source setup_release.sh Linux -5\n")
        script.write("echo 'sourcing done' \n")
        script.write(" export MKL_NUM_THREADS=1 \n")
        script.write(" export NUMEXPR_NUM_THREADS=1 \n")
        script.write(" export OMP_NUM_THREADS=1 \n")
        script.write(" export OPENBLAS_NUM_THREADS=1 \n")


    for iconf, val in enumerate(config):
        cmd_ = cmd(dbName, dbDir, dbExtens, val, outDir, ibatch, iconf, nproc)
        script.write(cmd_+" \n")

    if batch:
        script.write("EOF" + "\n")
    script.close()
    if batch:
        #os.system("sh "+scriptName)
        print('gone')

def prepareOut(dbName, confname, ibatch):
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

    id = '{}_{}_{}'.format(dbName, confname, ibatch)

    name_id = 'simulation_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'

    return dirScript, name_id, log, cwd


def config_sn(zmin, zmax, zstep, 
              x1_type='unique',x1_min=-2.0,x1_max=2.0,
              color_type='unique',color_min=0.2,color_max=0.4,
              z_type='random',daymax_type='random',daymax_step=2,
              nsn_factor=100,nsn_absolute=pd.DataFrame(),typeSN='faintSN',
              x1sigma=0,colorsigma=0):

    r = []
    for z in np.arange(zmin, zmax, zstep):
        z = np.round(z, 2)
        z_min = z
        z_max = z+np.round(zstep, 2)
        idx = np.abs(nsn_absolute['z']-z_max)<1.e-5
        nsn_abs = nsn_absolute[idx]['nsn_absolute'].to_list()[0]

        r.append((x1_type, x1_min, x1_max, color_type, color_min, color_max, z_type,
                  z_min, z_max, zstep, daymax_type, daymax_step, nsn_factor, nsn_abs, typeSN,x1sigma,colorsigma))

    res = pd.DataFrame(r, columns=['x1_type', 'x1_min', 'x1_max',
                                   'color_type', 'color_min', 'color_max',
                                   'z_type', 'z_min', 'z_max', 'z_step',
                                   'daymax_type', 'daymax_step', 'nsn_factor', 'nsn_absolute',
                                   'confName','x1sigma','colorsigma'])

    return res




def config_rec(nabs,nsnfactor,x1sigma,colorsigma):

    zmin = 0.0
    zmax = 1.2
    zstep = 0.02


    zvals = np.arange(zmin, zmax+zstep, zstep)
    nsn_abs = [-1]*len(zvals)
    
    df = pd.DataFrame(zvals.tolist(), columns=['z'])

    if 'faintSN'  in nabs.keys():
        df['nsn_absolute'] = nabs['faintSN']


        faintSN = config_sn(zmin, zmax=1.0, zstep=zstep, 
                            x1_type='unique',x1_min=-2.0,x1_max=2.0,
                            color_type='unique',color_min=0.2,color_max=0.4, 
                            z_type='random',daymax_type='random',daymax_step=2,
                            nsn_factor=nsnfactor['faintSN'],nsn_absolute=df,typeSN='faintSN',
                            x1sigma=x1sigma,colorsigma=colorsigma)

    if 'mediumSN'  in nabs.keys():

        df['nsn_absolute'] = nabs['mediumSN']
        mediumSN = config_sn(zmin, zmax, zstep, 
                             x1_type='unique',x1_min=0.0,x1_max=0.3,
                             color_type='unique',color_min=0.0,color_max=0.2, 
                             z_type='random',daymax_type='random',daymax_step=2,
                             nsn_factor=nsnfactor['mediumSN'],nsn_absolute=df,typeSN='mediumSN',
                             x1sigma=x1sigma,colorsigma=colorsigma)

    if 'mediumSN'  in nabs.keys():
        df['nsn_absolute'] = nabs['brightSN']
        brightSN = config_sn(zmin, zmax, zstep, 
                             x1_type='unique',x1_min=2.0,x1_max=0.3,
                             color_type='unique',color_min=-0.2,color_max=-0.2, 
                             z_type='random',daymax_type='random',daymax_step=2,
                             nsn_factor=nsnfactor['brightSN'],nsn_absolute=df,typeSN='brightSN',
                             x1sigma=x1sigma,colorsigma=colorsigma)

    
    if 'allSN'  in nabs.keys():

        df['nsn_absolute'] = nabs['allSN']
        ik = df['z']> 0.5
        if nabs['allSN']>0:
            df.loc[ik,'nsn_absolute'] = 2*nabs['allSN']

        allSN = config_sn(zmin, zmax, zstep, 
                          x1_type='random',x1_min=-3.0,x1_max=3.0,
                          color_type='random',color_min=-0.3,color_max=0.3, 
                          z_type='random',daymax_type='random',daymax_step=2,
                          nsn_factor=nsnfactor['allSN'],nsn_absolute=df,typeSN='allSN',
                          x1sigma=x1sigma,colorsigma=colorsigma)

    df_all = pd.DataFrame()
    if 'faintSN'  in nabs.keys():
        df_all = pd.concat((df_all, faintSN))
    if 'mediumSN'  in nabs.keys():
        df_all = pd.concat((df_all, mediumSN))
    if 'brightSN'  in nabs.keys():
        df_all = pd.concat((df_all, brightSN))
    if 'allSN'  in nabs.keys():
        df_all = pd.concat((df_all, allSN))


    return df_all.to_records(index=False)
    """
    r = []

    x1_type = 'unique'
    x1_min = -2.0
    x1_max = 2.0
    color_type = 'unique'
    color_min = 0.2
    color_max = 0.4
    #z_type = 'uniform'
    #daymax_type = 'uniform'
    z_type = 'random'
    daymax_type = 'random'
    daymax_step = 2
    nsn_factor = 100
    nsn_absolute = -1

    # this is for faintSN
    z_step = 0.1
    for z in np.arange(zmin, 1.0, z_step):
        z = np.round(z, 2)
        z_min = z
        z_max = z+np.round(z_step, 2)

        r.append((x1_type, x1_min, x1_max, color_type, color_min, color_max, z_type,
                  z_min, z_max, zstep, daymax_type, daymax_step, nsn_factor, nsn_absolute, 'faintSN'))

    for z in np.arange(zmin, zmax, zstep):
        zval = np.round(z, 2)
        x1_type = 'unique'
        x1_min = 0.0
        x1_max = 2.0
        color_type = 'unique'
        color_min = 0.0
        color_max = 0.4
        z_type = 'random'
        day_max = 'random'
        z_min = zval
        z_max = zval+np.round(zstep, 2)
        z_step = np.round(zstep, 2)
        nsn_factor_run = nsn_factor
        nsn_absolute_run = nsn_absolute
        r.append((x1_type, x1_min, x1_max, color_type, color_min, color_max, z_type,
                  z_min, z_max, z_step, daymax_type, daymax_step, nsn_factor, nsn_absolute, 'mediumSN'))

        z_type = 'random'
        x1_type = 'random'
        color_type = 'random'
        daymax_type = 'random'

        if z_max <= 0.5:
            nsn_absolute_run = 1000
        else:
            nsn_absolute_run = 2000
        r.append((x1_type, x1_min, x1_max, color_type, color_min, color_max, z_type,
                  z_min, z_max, z_step, daymax_type, daymax_step, nsn_factor_run, nsn_absolute_run, 'allSN'))

    res = np.rec.fromrecords(r, names=['x1_type', 'x1_min', 'x1_max',
                                       'color_type', 'color_min', 'color_max',
                                       'z_type', 'z_min', 'z_max', 'z_step',
                                       'daymax_type', 'daymax_step', 'nsn_factor', 'nsn_absolute',
                                       'confName'])
    """

    return res


def cmd(dbName, dbDir, dbExtens, config, outDir, ibatch, iconfig, nproc):

    errmod = 0
    error_model = 'error_model'
    if not errmod:
        error_model = '{}_{}'.format(380.0, 800.0)

    cmd = 'python run_scripts/simulation/run_simulation.py'
    cmd += ' --dbName {}'.format(dbName)
    cmd += '  --dbExtens {}'.format(dbExtens)
    cmd += ' --dbDir {}'.format(dbDir)
    cmd += ' --Observations_fieldtype Fake'
    cmd += ' --Observations_coadd 0'
    cmd += ' --RAmin -2.0'
    cmd += ' --RAmax 2.0'
    cmd += ' --radius 0.1'
    cmd += ' --nproc 1'

    for vv in ['x1', 'color', 'z']:
        cmd += ' --SN_{}_type {}' .format(vv, config['{}_type'.format(vv)])
    for vv in ['x1', 'color']:
        if config['{}_type'.format(vv)] == 'unique':
            cmd += ' --SN_{}_min {}'.format(vv, config['{}_min'.format(vv)])

    cmd += ' --SN_z_min {} --SN_z_max {} --SN_z_step {}'.format(
        np.round(config['z_min'], 2), np.round(config['z_max'], 2), np.round(config['z_step'], 2))
    cmd += ' --SN_daymax_type {} --SN_daymax_step {}'.format(
        config['daymax_type'], config['daymax_step'])
    cmd += ' --SN_NSNfactor {}'.format(config['nsn_factor'])
    cmd += ' --SN_NSNabsolute {}'.format(config['nsn_absolute'])
    cmd += ' --ProductionIDSimu Fakes_{}_{}_{}_{}_{}'.format(
        dbName, error_model, config['confName'], ibatch, iconfig)
    cmd += ' --Simulator_errorModel {}'.format(errmod)
    outputDir = '{}/{}'.format(outDir, dbName)
    cmd += ' --OutputSimu_directory {}'.format(outputDir)
    cmd += ' --SN_minRFphaseQual -15.'
    cmd += ' --SN_maxRFphaseQual 30.'
    cmd += ' --SN_ebvofMW 0.01'
    cmd += ' --Pixelisation_nside 64'
    cmd += ' --SN_modelPar_x1sigma {}'.format(config['x1sigma'])
    cmd += ' --SN_modelPar_colorsigma {}'.format(config['colorsigma'])

    # create outputDir here
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    print(cmd)
    return cmd


parser = OptionParser()

parser.add_option('--dbName', type='str', default='Fake_Obs',
                  help='dbName to process  [%default]')
parser.add_option('--dbDir', type='str',
                  default='Fake_Observations', help='db dir [%default]')
parser.add_option('--dbExtens', type='str', default='npy',
                  help='db extension [%default]')
parser.add_option('--outDir', type='str',
                  default='Fakes/Simu', help='output directory [%default]')
parser.add_option('--batch', type=int, default=1,
                  help='running in batch mode[%default]')
parser.add_option('--nproc', type=int, default=8,
                  help='number of proc [%default]')
parser.add_option('--snTypes', type=str,
                  default='faintSN,allSN', help='SN types to process [%default]')
parser.add_option('--nabs', type=str,
                  default='-1,1500',help='absolute number for production [%default]')
parser.add_option('--nsnfactor', type=str,
                  default='100,100',help='factor for nsn production [%default]')
parser.add_option("--x1sigma", type=int, default=0,help="shift of x1 parameter distribution[%default]")
parser.add_option("--colorsigma", type=int, default=0,help="shift of color parameter distribution[%default]")

opts, args = parser.parse_args()

print('Start processing...')

snTypes = opts.snTypes.split(',')
nabs = list(map(int,opts.nabs.split(',')))
nsnfactor = list(map(int,opts.nsnfactor.split(',')))

if len(snTypes) != len(nabs):
    print('problem - check snTypes and nabs')

if len(snTypes) != len(nsnfactor):
    print('problem - check snTypes and nsnfactor')

assert(len(snTypes) == len(nabs))
assert(len(snTypes) == len(nsnfactor))

nabs = dict(zip(snTypes,nabs))
nsnfactor = dict(zip(snTypes,nsnfactor))
x1sigma = opts.x1sigma
colorsigma = opts.colorsigma

print('booo',nabs,nsnfactor)
process(opts.dbName, opts.dbDir, opts.dbExtens,
        opts.outDir, opts.nproc, opts.batch, snTypes,nabs,nsnfactor,x1sigma,colorsigma)
