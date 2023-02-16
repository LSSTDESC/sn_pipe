import os
import numpy as np
from optparse import OptionParser

def createDirs(dirLC, dirTemplate):

    # create output directory
    if not os.path.isdir(dirLC):
        os.makedirs(dirLC)

    # List of requested directories
    fake_obs_yaml = '{}/fake_obs_yaml'.format(dirLC)
    fake_obs_data = '{}/fake_obs_data'.format(dirLC)
    fake_simu_yaml = '{}/fake_simu_yaml'.format(dirLC)
    fake_simu_data = '{}/fake_simu_data'.format(dirLC)

    # create these directory if necessary
    for vv in [fake_obs_yaml, fake_obs_data, fake_simu_yaml, fake_simu_data]:
        if not os.path.isdir(vv):
            os.makedirs(vv)

    # create output directory
    if not os.path.isdir(dirTemplate):
        os.makedirs(dirTemplate) 

    

def addoption(cmd, name, val):
    cmd += ' --{} {}'.format(name, val)
    return cmd


def process(x1, color, sn_type='SN_Ia',sn_model='salt2-extended',diff_flux=1,nproc=8, zmin=0.01, zmax=1.2, zstep=0.01, ebvofMW=-1.,
          bluecutoff=380., redcutoff=800.,error_model=0,
          outDirLC='', outDirTemplates='', what='simu',mode='batch'):

    cwd = os.getcwd()
    dirScript = cwd + "/scripts"

    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)

    cutoff = '{}_{}'.format(bluecutoff, redcutoff)
    if error_model>0:
        cutoff = 'error_model'
    id = '{}_{}_{}_ebvofMW_{}'.format(
        x1, color, cutoff,ebvofMW)
    name_id = 'template_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'

    qsub = 'qsub -P P_lsst -l sps=1,ct=2:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
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

    cmd = 'python run_scripts/templates/run_template_LC.py'
    cmd = addoption(cmd, 'x1', x1)
    cmd = addoption(cmd, 'color', color)
    cmd = addoption(cmd, 'zmin', zmin)
    cmd = addoption(cmd, 'zmax', zmax)
    cmd = addoption(cmd, 'zstep', zstep)
    cmd = addoption(cmd, 'nproc', nproc)
    cmd = addoption(cmd, 'outDir', outDirLC)
    cmd = addoption(cmd, 'ebvofMW', ebvofMW)
    cmd = addoption(cmd, 'bluecutoff', bluecutoff)
    cmd = addoption(cmd, 'redcutoff', redcutoff)
    cmd = addoption(cmd, 'error_model', error_model)
    cmd = addoption(cmd, 'sn_type',sn_type)
    cmd = addoption(cmd, 'sn_model', sn_model)
    cmd = addoption(cmd, 'diff_flux', diff_flux)

    if what == 'simu':
        print(cmd)
        if mode == 'batch':
            script.write(cmd+"\n")
            script.write("EOF" + "\n")
            script.close()
            os.system("sh "+scriptName)
        else:
            os.system(cmd)
            

    # stack produced LCs
    cmd = 'python run_scripts/templates/run_template_vstack.py'
    cmd = addoption(cmd, 'x1', x1)
    cmd = addoption(cmd, 'color', color)
    cmd = addoption(cmd, 'lcDir', '{}/fake_simu_data'.format(outDirLC))
    cmd = addoption(cmd, 'outDir', outDirTemplates)
    cmd = addoption(cmd, 'bluecutoff', bluecutoff)
    cmd = addoption(cmd, 'redcutoff', redcutoff)
    cmd = addoption(cmd, 'ebvofMW', ebvofMW)
    cmd = addoption(cmd, 'error_model', error_model)   
    cmd = addoption(cmd, 'sn_type',sn_type)
    cmd = addoption(cmd, 'sn_model', sn_model)
    cmd = addoption(cmd, 'diff_flux', diff_flux)

    if what == 'vstack':
        print(cmd)
        os.system(cmd)


parser = OptionParser()

parser.add_option("--action", type="str", default='simu',
                  help="what to do: simu or vstack[%default]")
parser.add_option("--mode", type="str", default='batch',
                  help="how to run: batch or interactive [%default]")
parser.add_option("--x1", type=float, default=-2.0,
                  help="SN x1[%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="SN color[%default]")
parser.add_option("--sn_type", type=str, default='SN_Ia',
                  help="SN type [%default]")
parser.add_option("--sn_model", type=str, default='salt2-extended',
                  help="SN model [%default]")
parser.add_option("--zmax", type=float, default=1.01,
                  help="SN max redshift[%default]")
parser.add_option("--zmin", type=float, default=0.01,
                  help="SN min redshift[%default]")
parser.add_option("--bluecutoff", type=float, default=380.0,
                  help="blue cutoff[%default]")
parser.add_option("--redcutoff", type=float, default=800.0,
                  help="red cutoff[%default]")
parser.add_option("--ebv", type=float, default=0.0,
                  help="E(B-V)[%default]")
parser.add_option("--error_model", type=int, default=1,
                  help="error model for SN LC estimation[%default]")
parser.add_option("--diff_flux", type=int, default=1,
                  help="to make simulations with simulator param variation [%default]")


opts, args = parser.parse_args()


ebvofMW = np.round(opts.ebv,2)
 
cutoff = '{}_{}'.format(opts.bluecutoff, opts.redcutoff)
if opts.error_model>0:
    cutoff = 'error_model'
 
fname = '{}_{}'.format(opts.sn_type,opts.sn_model)

if 'salt2' in opts.sn_model:
    fname = '{}_{}'.format(opts.x1,opts.color)

outDirLC = '/sps/lsst/users/gris/fakes_for_templates_{}'.format(fname)
outDirTemplates = '/sps/lsst/users/gris/Template_LC_{}'.format(fname)

outDirLC_ebv = '{}_{}_ebvofMW_{}'.format(outDirLC, cutoff,ebvofMW)
outDirTemplates_ebv = '{}_{}_ebvofMW_{}'.format(outDirTemplates, cutoff,ebvofMW)

# create requested output directories
createDirs(outDirLC_ebv,outDirTemplates_ebv)
process(opts.x1, opts.color,opts.sn_type,opts.sn_model,opts.diff_flux,
        zmin=opts.zmin, zmax=opts.zmax,
        ebvofMW=ebvofMW,
        bluecutoff=opts.bluecutoff,
        redcutoff=opts.redcutoff,
        error_model=opts.error_model,
        outDirLC=outDirLC_ebv, 
        outDirTemplates=outDirTemplates_ebv, 
        what=opts.action,mode=opts.mode)
