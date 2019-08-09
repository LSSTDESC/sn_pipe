import os
import numpy as np

def batch(dbDir,dbName,scriptref,band,nproc):
    cwd = os.getcwd()
    dirScript= cwd + "/scripts"

    if not os.path.isdir(dirScript) :
        os.makedirs(dirScript)
    
    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog) :
        os.makedirs(dirLog)    
    
    
    id='{}_{}'.format(dbName,band)
    name_id='metric_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'


    qsub = 'qsub -P P_lsst -l sps=1,ct=05:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(log,nproc)
    #qsub = "qsub -P P_lsst -l sps=1,ct=05:00:00,h_vmem=16G -j y -o "+ log + " <<EOF"
    scriptName = dirScript+'/'+name_id+'.sh'


    script = open(scriptName,"w")
    script.write(qsub + "\n")
    script.write("#!/usr/local/bin/bash\n")
    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh CCIN2P3\n")
    script.write("echo 'sourcing done' \n")

    cmd='python {}.py --dbDir {} --dbName {} --nproc {} --band {}'.format(scriptref,dbDir,dbName,nproc,band)
    script.write(cmd+" \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)

"""
dbDir ='/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db/2018-06-WPC' 

dbNames=['kraken_2026','kraken_2042','kraken_2035','kraken_2044']
dbNames = ['kraken_2026','kraken_2042','kraken_2035','kraken_2044','colossus_2667','pontus_2489','pontus_2002','mothra_2049','nexus_2097']

for dbName in dbNames:
    batch(dbDir,dbName,'run_metric',8)

"""
bands = 'griz'
dbDir = '/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db'
dbNames = ['alt_sched','alt_sched_rolling','rolling_10yrs','rolling_mix_10yrs']
dbNames += ['kraken_2026','kraken_2042','kraken_2035','kraken_2044','colossus_2667','pontus_2489','pontus_2002','mothra_2049','nexus_2097']
dbNames += ['baseline_1exp_nopairs_10yrs','baseline_1exp_pairsame_10yrs','baseline_1exp_pairsmix_10yrs','baseline_2exp_pairsame_10yrs',
          'baseline_2exp_pairsmix_10yrs','ddf_0.23deg_1exp_pairsmix_10yrs','ddf_0.70deg_1exp_pairsmix_10yrs',
          'ddf_pn_0.23deg_1exp_pairsmix_10yrs','ddf_pn_0.70deg_1exp_pairsmix_10yrs','exptime_1exp_pairsmix_10yrs','baseline10yrs',
          'big_sky10yrs','big_sky_nouiy10yrs','gp_heavy10yrs','newA10yrs','newB10yrs','roll_mod2_mixed_10yrs',
          'roll_mod3_mixed_10yrs','roll_mod6_mixed_10yrs','simple_roll_mod10_mixed_10yrs','simple_roll_mod2_mixed_10yrs',
          'simple_roll_mod3_mixed_10yrs','simple_roll_mod5_mixed_10yrs','twilight_1s10yrs',
          'altsched_1exp_pairsmix_10yrs','rotator_1exp_pairsmix_10yrs','hyak_baseline_1exp_nopairs_10yrs',
          'hyak_baseline_1exp_pairsame_10yrs']


dbNames = ['very_alt2_rm5illum20_10yrs','very_alt2_rm5illum40_10yrs','very_alt3_rm5illum20_10yrs','very_alt3_rm5illum40_10yrs','very_alt10yrs','very_alt2_rm5illum25_10yrs','very_alt2_rm5illum50_10yrs','very_alt3_rm5illum25_10yrs','very_alt3_rm5illum50_10yrs','very_alt2_rm5illum15_10yrs','very_alt2_rm5illum30_10yrs','very_alt3_rm5illum15_10yrs','very_alt3_rm5illum30_10yrs','very_alt_rm510yrs','noddf_1exp_pairsame_10yrs','desc_ddf_pn_0.70deg_1exp_pairsmix_10yrs','fc1exp_pairsmix_ilim30_10yrs','fc1exp_pairsmix_ilim60_10yrs',
'fc1exp_pairsmix_ilim15_10yrs','stuck_rolling10yrs','shortt_2ns_1ext_pairsmix_10yrs','shortt_2ns_5ext_pairsmix_10yrs','shortt_5ns_5ext_pairsmix_10yrs','shortt_5ns_1ext_pairsmix_10yrs',
'simple_roll_mod2_mixed_10yrs','roll_mod2_sdf0.2mixed_10yrs','simple_roll_mod3_sdf0.2mixed_10yrs',
'roll_mod2_sdf0.1mixed_10yrs','roll_mod3_sdf0.2mixed_10yrs',
'roll_mod3_sdf0.1mixed_10yrs','simple_roll_mod5_sdf0.2mixed_10yrs',
'roll_mod6_sdf0.2mixed_10yrs','roll_mod6_sdf0.1mixed_10yrs',
'simple_roll_mod10_sdf0.2mixed_10yrs','roll_mod2_sdf0.10mixed_10yrs',
'roll_mod2_sdf0.05mixed_10yrs','simple_roll_mod2_sdf0.20mixed_10yrs',
'roll_mod3_sdf0.05mixed_10yrs','roll_mod2_sdf0.20mixed_10yrs',
'roll_mod3_sdf0.20mixed_10yrs','simple_roll_mod3_sdf0.20mixed_10yrs',
'roll_mod3_sdf0.10mixed_10yrs','roll_mod6_sdf0.05mixed_10yrs',
'roll_mod6_sdf0.20mixed_10yrs','roll_mod6_sdf0.10mixed_10yrs',
'simple_roll_mod10_sdf0.20mixed_10yrs']

#dbNames = ['weather_0.20c_10yrs','weather_0.60c_10yrs','weather_0.70c_10yrs','weather_1.10c_10yrs',
#'weather_0.40c_10yrs','weather_0.90c_10yrs','weather_0.30c_10yrs','weather_0.80c_10yrs','weather_0.10c_10yrs']

print(len(dbNames))
#dbDir = '/sps/lsst/cadence/LSST_SN_PhG/cadence_db/opsim_new'
for dbName in dbNames:
    for band in bands:
        batch(dbDir,dbName,'run_scripts/run_metrics_fromnpy',band,8)
    #batch(dbDir,dbName,'run_scripts/run_metrics_fromnpy','all',8)
    
