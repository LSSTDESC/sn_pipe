import os
import numpy as np


def batch(x1, color, nproc=8, zmax=1.2):

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
    script.write(" source setup_release.sh CCIN2P3\n")
    script.write(" source export.sh \n")
    script.write("echo 'sourcing done' \n")
    """
    script.write("export PYTHONPATH=sn_tools:$PYTHONPATH \n")
    script.write("export PYTHONPATH=sn_metrics:$PYTHONPATH \n")
    script.write("export PYTHONPATH=sn_stackers:$PYTHONPATH \n")
    """
    script.write("echo $PYTHONPATH \n")

    cmd = 'python run_scripts/templates/run_simulation_template.py --simul 1 --lcdiff 0 --x1 {} --color {} --nproc {} --zmax {}'.format(
        x1, color, nproc, zmax)
    script.write(cmd+" \n")
    cmd = 'python run_scripts/templates/run_simulation_template.py --simul 0 --lcdiff 1 --x1 {} --color {}'.format(
        x1, color)
    script.write(cmd+" \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)


x1_colors = [(-2.0, -0.2), (-2.0, 0.0), (-2.0, 0.2),
             (0.0, -0.2), (0.0, 0.0), (0.0, 0.2),
             (2.0, -0.2), (2.0, 0.0), (2.0, 0.2)]

zmax = [1.1, 1.1, 0.8, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]

zmax_dict = dict(zip(x1_colors, zmax))

for (x1, color) in x1_colors:
    batch(x1, color, zmax=zmax_dict[(x1, color)])
