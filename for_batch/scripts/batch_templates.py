import os
import numpy as np
from optparse import OptionParser


def addoption(cmd, name, val):
    cmd += ' --{} {}'.format(name, val)
    return cmd


def process(x1, color, nproc=8, zmin=0.01, zmax=1.2, zstep=0.01, ebvofMW=-1.,
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
    if what == 'vstack':
        print(cmd)
        os.system(cmd)


parser = OptionParser()

parser.add_option("--action", type="str", default='simu',
                  help="what to do: simu or vstack[%default]")
parser.add_option("--mode", type="str", default='batch',
                  help="how to run: batch or interactive [%default]")


opts, args = parser.parse_args()

x1_colors = [(-2.0, -0.2), (-2.0, 0.0), (-2.0, 0.2),
             (0.0, -0.2), (0.0, 0.0), (0.0, 0.2),
             (2.0, -0.2), (2.0, 0.0), (2.0, 0.2)]

zmax = [1.1, 1.1, 0.8, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
x1_colors = [(-2.0, 0.2), (0.0, 0.0)]
zmax = [0.8, 1.2]

x1_colors = [(-2.0,0.2)]
zmax = [0.8]
zmax_dict = dict(zip(x1_colors, zmax))

outDirLC = '/sps/lsst/users/gris/fakes_for_templates'
outDirTemplates = '/sps/lsst/users/gris/Template_LC'
bluecutoff = 380.
redcutoff = 800.
#ebvs = np.arange(0.0, 0.40, 0.01)
ebvs= [0.0]
error_model = 1
"""
outDirLC = '{}_{}_{}'.format(
    outDirLC, bluecutoff, redcutoff)
outDirTemplates = '{}_{}_{}'.format(
    outDirTemplates, bluecutoff, redcutoff)
"""
cutoff = '{}_{}'.format(bluecutoff, redcutoff)
if error_model>0:
    cutoff = 'error_model'
for (x1, color) in x1_colors:
    for ebvofMW in ebvs:
        ebvofMW = np.round(ebvofMW,2)
        outDirLC_ebv = '{}_{}_ebvofMW_{}'.format(outDirLC, cutoff,ebvofMW)
        outDirTemplates_ebv = '{}_{}_ebvofMW_{}'.format(outDirTemplates, cutoff,ebvofMW)
        

        process(x1, color, zmax=zmax_dict[(x1, color)],
                ebvofMW=ebvofMW,
                bluecutoff=bluecutoff,
                redcutoff=redcutoff,
                error_model=error_model,
                outDirLC=outDirLC_ebv, 
                outDirTemplates=outDirTemplates_ebv, 
                what=opts.action,mode=opts.mode)
        
