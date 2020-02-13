import os
import numpy as np


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
    script.write(" source setup_release.sh CCIN2P3\n")
    script.write("echo 'sourcing done' \n")
    script.write(" source export.sh CCIN2P3\n")

    cmd = 'python {}.py --dbDir {} --dbName {} --dbExtens {} --nproc {}'.format(
        scriptref, dbDir, dbName, nproc, dbExtens)
    script.write(cmd+" \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)


dbDir = '/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db'
dbNames = ['alt_sched', 'alt_sched_rolling', 'kraken_2026']
"""
dbNames += ['rolling_10yrs','rolling_mix_10yrs']
dbNames += ['kraken_2042','kraken_2035','kraken_2044','colossus_2667','pontus_2489','pontus_2002','mothra_2049','nexus_2097']
dbNames += ['baseline_1exp_nopairs_10yrs','baseline_1exp_pairsame_10yrs','baseline_1exp_pairsmix_10yrs','baseline_2exp_pairsame_10yrs',
          'baseline_2exp_pairsmix_10yrs','ddf_0.23deg_1exp_pairsmix_10yrs','ddf_0.70deg_1exp_pairsmix_10yrs',
          'ddf_pn_0.23deg_1exp_pairsmix_10yrs','ddf_pn_0.70deg_1exp_pairsmix_10yrs','exptime_1exp_pairsmix_10yrs','baseline10yrs',
          'big_sky10yrs','big_sky_nouiy10yrs','gp_heavy10yrs','newA10yrs','newB10yrs','roll_mod2_mixed_10yrs',
          'roll_mod3_mixed_10yrs','roll_mod6_mixed_10yrs','simple_roll_mod10_mixed_10yrs','simple_roll_mod2_mixed_10yrs',
          'simple_roll_mod3_mixed_10yrs','simple_roll_mod5_mixed_10yrs','twilight_1s10yrs',
          'altsched_1exp_pairsmix_10yrs','rotator_1exp_pairsmix_10yrs','hyak_baseline_1exp_nopairs_10yrs',
          'hyak_baseline_1exp_pairsame_10yrs']

for dbName in dbNames:
    batch(dbDir,dbName,'run_scripts/run_global_metric',8)


dbDir = '/sps/lsst/cadence/LSST_SN_PhG/cadence_db/opsim_npy'
dbNames = ['dec_1exp_pairsmix_10yrs']
for dbName in dbNames:
    batch(dbDir,dbName,'run_scripts/run_global_metric',8)

"""
dbDir = '/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.3'
dbNames = ['baseline_2snap_v1.3_10yrs', 'baseline_v1.3_10yrs', 'baseline_nomix_v1.3_10yrs', 'agnddf_illum60_v1.3_10yrs', 'agnddf_illum30_v1.3_10yrs', 'agnddf_illum10_v1.3_10yrs', 'agnddf_illum15_v1.3_10yrs', 'descddf_illum60_v1.3_10yrs',
           'descddf_illum30_v1.3_10yrs', 'descddf_illum7_v1.3_10yrs', 'descddf_illum15_v1.3_10yrs', 'descddf_illum10_v1.3_10yrs', 'descddf_illum3_v1.3_10yrs', 'euclid_ddf_v1.3_10yrs', 'descddf_illum4_v1.3_10yrs', 'descddf_illum5_v1.3_10yrs', 'euclid_ddf_v1.3_10yrs']

dbExtens = 'db'

for dbName in dbNames:
    batch(dbDir, dbName, dbExtens, 'run_scripts/metrics/run_global_metric', 8)
