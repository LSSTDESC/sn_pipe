import os
import numpy as np
from optparse import OptionParser
import pandas as pd


def batch(dbDir, dbName, dbExtens, scriptref, nproc):
    cwd = os.getcwd()
    dirScript = cwd + "/scripts"

    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)

    id = '{}_global'.format(dbName)
    name_id = 'metric_Global{}'.format(id)
    log = dirLog + '/'+name_id+'.log'

    qsub = 'qsub -P P_lsst -l sps=1,ct=02:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
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

    cmd = 'python {}.py --dbDir {} --dbName {} --dbExtens {} --nproc {}'.format(
        scriptref, dbDir, dbName, dbExtens, nproc)
    script.write(cmd+" \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)


parser = OptionParser()

parser.add_option("--fileList", type="str", default='global.csv',
                  help="what to do: simu or vstack[%default]")

opts, args = parser.parse_args()

toproc = pd.read_csv(opts.fileList, comment='#')

for index, row in toproc.iterrows():
    print(row['dbDir'], row['dbName'], row['dbExtens'])
    batch(row['dbDir'], row['dbName'], row['dbExtens'],
          'run_scripts/metrics/run_global_metric', 8)
