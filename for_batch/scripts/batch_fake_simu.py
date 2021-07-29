from optparse import OptionParser
import pandas as pd
import os
import numpy as np
import itertools


def process(dbName, dbDir, dbExtens, outDir, nproc=8, mode='batch', snTypes=['faintSN', 'allSN']):

    if mode == 'batch':
        batch(dbName, dbDir, dbExtens, outDir, snTypes, nproc)
    else:
        config = config_rec()
        confNames = snTypes
        for cf in confNames:
            idx = config['confName'] == cf
            sel = config[idx]
            print('alors', sel)
            spl = np.array_split(sel, 5)
            cmd_ = cmd(dbName, dbDir, dbExtens, outDir, 0, nproc)
            print(cmd_)
            # os.system(cmd_)


def batch(dbName, dbDir, dbExtens, outDir, snTypes, nproc=8):
    # get config for the run
    config = config_rec()

    #confNames = ['faintSN','allSN']
    #confNames = ['faintSN']
    confNames = snTypes
    for cf in confNames:
        idx = config['confName'] == cf
        sel = config[idx]
        spl = np.array_split(sel, np.min([1, len(sel)]))
        for ibatch, vv in enumerate(spl):
            batch_indiv(dbName, dbDir, dbExtens, outDir, cf, vv, ibatch, nproc)


def batch_indiv(dbName, dbDir, dbExtens, outDir, confname, config, ibatch, nproc=8):

    dirScript, name_id, log, cwd = prepareOut(
        dbName, confname, ibatch)
    # qsub command
    qsub = 'qsub -P P_lsst -l sps=1,ct=3:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
        log, nproc)

    scriptName = dirScript+'/'+name_id+'.sh'

    # fill the script
    script = open(scriptName, "w")
    script.write(qsub + "\n")
    script.write("#!/bin/env bash\n")
    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh Linux\n")
    script.write("echo 'sourcing done' \n")

    for iconf, val in enumerate(config):
        cmd_ = cmd(dbName, dbDir, dbExtens, val, outDir, ibatch, iconf, nproc)
        script.write(cmd_+" \n")

    script.write("EOF" + "\n")
    script.close()
    #os.system("sh "+scriptName)


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
    #z_type = 'uniform'
    #daymax_type = 'uniform'
    z_type = 'random'
    daymax_type = 'random'
    daymax_step = 2
    nsn_factor = 100

    # this is for faintSN
    z_step = 0.1
    for z in np.arange(zmin, 1.0, z_step):
        z = np.round(z, 2)
        z_min = z
        z_max = z+np.round(z_step, 2)
        r.append((x1_type, x1_min, x1_max, color_type, color_min, color_max, z_type,
                  z_min, z_max, zstep, daymax_type, daymax_step, nsn_factor, 'faintSN'))

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
        nsn_factor = 50

        r.append((x1_type, x1_min, x1_max, color_type, color_min, color_max, z_type,
                  z_min, z_max, z_step, daymax_type, daymax_step, nsn_factor, 'mediumSN'))

        z_type = 'random'
        x1_type = 'random'
        color_type = 'random'
        daymax_type = 'random'

        r.append((x1_type, x1_min, x1_max, color_type, color_min, color_max, z_type,
                  z_min, z_max, z_step, daymax_type, daymax_step, nsn_factor, 'allSN'))

    res = np.rec.fromrecords(r, names=['x1_type', 'x1_min', 'x1_max',
                                       'color_type', 'color_min', 'color_max',
                                       'z_type', 'z_min', 'z_max', 'z_step',
                                       'daymax_type', 'daymax_step', 'nsn_factor',
                                       'confName'])

    return res


def get_config_deprecated():
    config = {}

    config['faintSN'] = {}
    config['allSN'] = {}

    zmin = 0.0
    zmax = 1.2
    zstep = 0.05

    for z in np.arange(zmin, zmax, zstep):
        zval = np.round(z, 2)
        config['allSN'][zval] = {}
        config['faintSN'][zval] = {}

        for vv in ['x1', 'color', 'z']:
            config['allSN'][zval][vv] = {}
            config['allSN'][zval][vv]['type'] = 'random'
            config['allSN'][zval][vv]['min'] = zval
            config['allSN'][zval][vv]['max'] = zval+zstep

        for vv in ['x1', 'color', 'z']:
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
    cmd += ' --ProductionIDSimu Fakes_{}_{}_{}_{}_{}'.format(
        dbName, error_model, config['confName'], ibatch, iconfig)
    cmd += ' --Simulator_errorModel {}'.format(errmod)
    outputDir = '{}/{}'.format(outDir, dbName)
    cmd += ' --OutputSimu_directory {}'.format(outputDir)
    cmd += ' --SN_minRFphaseQual -15.'
    cmd += ' --SN_maxRFphaseQual 30.'
    cmd += ' --SN_ebvofMW 0'
    cmd += ' --Pixelisation_nside 64'

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
parser.add_option('--mode', type='str', default='batch',
                  help='running mode batch/interactive[%default]')
parser.add_option('--nproc', type=int, default=8,
                  help='number of proc [%default]')
parser.add_option('--snTypes', type=str,
                  default='faintSN,allSN', help='SN types to process')

opts, args = parser.parse_args()

print('Start processing...')

snTypes = opts.snTypes.split(',')

process(opts.dbName, opts.dbDir, opts.dbExtens,
        opts.outDir, opts.nproc, opts.mode, snTypes)
