import os
import numpy as np

def addoption(cmd, name, val):
    cmd += ' --{} {}'.format(name, val)
    return cmd



def batch(x1, color, nproc=8, zmin=0.01,zmax=1.2,zstep=0.01,outDirLC='',outDirTemplates='',what='simu'):

    cwd = os.getcwd()
    dirScript = cwd + "/scripts"

    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)

    id = '{}_{}'.format(x1, color)
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
    if what == 'simu':
        print(cmd)
        script.write(cmd+"\n")



    # stack produced LCs
    cmd = 'python run_scripts/templates/run_template_vstack.py'
    cmd = addoption(cmd, 'x1', x1)
    cmd = addoption(cmd, 'color', color)
    cmd = addoption(cmd, 'lcDir', '{}/fake_simu_data'.format(outDirLC))
    cmd = addoption(cmd, 'outDir', outDirTemplates)
    if what == 'vstack':
        print(cmd)
        script.write(cmd+" \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)


x1_colors = [(-2.0, -0.2), (-2.0, 0.0), (-2.0, 0.2),
             (0.0, -0.2), (0.0, 0.0), (0.0, 0.2),
             (2.0, -0.2), (2.0, 0.0), (2.0, 0.2)]

zmax = [1.1, 1.1, 0.8, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]

zmax_dict = dict(zip(x1_colors, zmax))

outDirLC = '/sps/lsst/users/gris/fakes_for_templates'
outDirTemplates = '/sps/lsst/users/gris/Template_LC'

what='simu'
for (x1, color) in x1_colors:
    batch(x1, color, zmax=zmax_dict[(x1, color)],
          outDirLC=outDirLC,outDirTemplates=outDirTemplates,what=what)
    break
