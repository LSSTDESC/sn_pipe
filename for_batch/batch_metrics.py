import os
import numpy as np
from optparse import OptionParser

def batch(dbDir,dbName,scriptref,nside,simuType,outDir,nprocprog,nproccomp,fieldType,saveData,metric,coadd):

    cwd = os.getcwd()
    dirScript= cwd + "/scripts"

    if not os.path.isdir(dirScript) :
        os.makedirs(dirScript)
    
    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog) :
        os.makedirs(dirLog)    
    
    
    id='{}_{}_{}_{}'.format(dbName,nside,fieldType,metric)
    name_id='metric_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'


    qsub = 'qsub -P P_lsst -l sps=1,ct=10:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(log,nproccomp)
    #qsub = "qsub -P P_lsst -l sps=1,ct=05:00:00,h_vmem=16G -j y -o "+ log + " <<EOF"
    scriptName = dirScript+'/'+name_id+'.sh'


    script = open(scriptName,"w")
    script.write(qsub + "\n")
    script.write("#!/usr/local/bin/bash\n")
    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh CCIN2P3\n")
    script.write(" source export.sh CCIN2P3\n")
    script.write("echo 'sourcing done' \n")
    """
    script.write("export PYTHONPATH=sn_tools:$PYTHONPATH \n")
    script.write("export PYTHONPATH=sn_metrics:$PYTHONPATH \n")
    script.write("export PYTHONPATH=sn_stackers:$PYTHONPATH \n")
    """
    script.write("echo $PYTHONPATH \n")

    cmd = 'python {}.py --dbDir {} --dbName {}'.format(scriptref,dbDir,dbName)
    cmd += ' --nproc {} --nside {} --simuType {}'.format(nprocprog,nside,simuType)
    cmd += ' --outDir {}'.format(outDir)
    cmd += ' --fieldType {}'.format(fieldType)
    cmd += ' --saveData {}'.format(saveData)
    cmd += ' --metric {}'.format(metric)
    cmd += ' --coadd {}'.format(coadd)
    cmd += ' --nodither {}'.format(nodither)
    script.write(cmd+" \n")
    script.write("EOF" + "\n")
    script.close()
    #os.system("sh "+scriptName)

def batch_new(dbDir,dbExtens,scriptref,outDir,nproccomp,
              saveData,metric,toprocess,nodither,
              RA_min=0.0,RA_max=360.0,Dec_min=-1.0,Dec_max=-1.0):

    cwd = os.getcwd()
    dirScript= cwd + "/scripts"

    if not os.path.isdir(dirScript) :
        os.makedirs(dirScript)
    
    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog) :
        os.makedirs(dirLog)    
    
    dbName = toprocess['dbName'][0].decode()
    nside = toprocess['nside'][0]
    fieldType = toprocess['fieldType'][0].decode()
    id='{}_{}_{}_{}{}_{}_{}_{}_{}'.format(dbName,nside,fieldType,metric,nodither,RA_min,RA_max,Dec_min,Dec_max) 

    name_id='metric_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'


    qsub = 'qsub -P P_lsst -l sps=1,ct=24:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(log,nproccomp)
    #qsub = "qsub -P P_lsst -l sps=1,ct=05:00:00,h_vmem=16G -j y -o "+ log + " <<EOF"
    scriptName = dirScript+'/'+name_id+'.sh'


    script = open(scriptName,"w")
    script.write(qsub + "\n")
    script.write("#!/usr/local/bin/bash\n")
    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh CCIN2P3\n")
    script.write(" source export.sh CCIN2P3\n")
    script.write("echo 'sourcing done' \n")
    """
    script.write("export PYTHONPATH=sn_tools:$PYTHONPATH \n")
    script.write("export PYTHONPATH=sn_metrics:$PYTHONPATH \n")
    script.write("export PYTHONPATH=sn_stackers:$PYTHONPATH \n")
    """
    script.write("echo $PYTHONPATH \n")

    for proc in toprocess:
        cmd_ = batch_cmd(scriptref,dbDir,dbExtens,outDir,
                         saveData,metric,proc,nodither,
                         RA_min,RA_max,Dec_min,Dec_max)

        
        script.write(cmd_+" \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)

def batch_cmd(scriptref,dbDir,dbExtens,outDir,
              saveData,metric,proc,nodither,
              RA_min,RA_max,Dec_min,Dec_max):

    cmd = 'python {}.py --dbDir {} --dbName {} --dbExtens {}'.format(scriptref,dbDir,proc['dbName'].decode(),dbExtens)
    cmd += ' --nproc {} --nside {} --simuType {}'.format(proc['nproc'],proc['nside'],proc['simuType'])
    cmd += ' --outDir {}'.format(outDir)
    cmd += ' --fieldType {}'.format(proc['fieldType'].decode())
    cmd += ' --saveData {}'.format(saveData)
    cmd += ' --metric {}'.format(metric)
    cmd += ' --coadd {}'.format(proc['coadd'])
    if nodither != '':
        cmd += ' --nodither {}'.format(nodither)

    cmd += ' --ramin {}'.format(RA_min)
    cmd += ' --ramax {}'.format(RA_max)
    cmd += ' --decmin {}'.format(Dec_min)
    cmd += ' --decmax {}'.format(Dec_max)

    return cmd


"""
if fieldType =='DD':
    dbNames = ['kraken_2026','ddf_0.23deg_1exp_pairsmix_10yrs','ddf_0.70deg_1exp_pairsmix_10yrs','ddf_pn_0.23deg_1exp_pairsmix_10yrs','ddf_pn_0.70deg_1exp_pairsmix_10yrs']
    simuType = [0,1,1,1,1]
    nproc=5
    nside = 128

if fieldType =='WFD':
    dbNames = ['kraken_2026','alt_sched','altsched_good_weather','alt_sched_rolling','baseline_1exp_nopairs_10yrs']
    simuType = [1,2,2,2,1]
    #dbNames = ['kraken_2026']
    #dbNames = ['altsched_good_weather']
    #'baseline_1exp_pairsame_10yrs','baseline_1exp_pairsmix_10yrs','baseline_2exp_pairsame_10yrs','baseline_2exp_pairsmix_10yrs','roll_mod2_sdf0.2mixed_10yrs']
    #dbNames = ['alt_sched_rolling', 'kraken_2026','rolling_10yrs_opsim']
    #simuType = [2,1,0]
    dbNames = ['colossus_2667']
    simuType = [2]
    
    #simuType = [1]
    #simuType = [2]
    nproc = 8
    nside = 64
    
    dbNames = dbNames_oswg_paper
    simuType = simuType_oswg_paper
    
"""
    
parser = OptionParser()

parser.add_option("--dbList", type="str", default='WFD.txt',
                  help="dbList to process  [%default]")
parser.add_option("--metricName", type="str", default='SNR',
                  help="metric to process  [%default]")
parser.add_option("--dbDir", type="str", default='', help="db dir [%default]")
parser.add_option("--dbExtens", type="str", default='npy', help="db extension [%default]")
parser.add_option("--nodither", type="str", default='', help="db extension [%default]")
parser.add_option("--splitSky", type="int", default=0, help="db extension [%default]")

opts, args = parser.parse_args()

print('Start processing...')

dbList = opts.dbList
metricName = opts.metricName
dbDir = opts.dbDir
if dbDir == '':
    dbDir = '/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db'
dbExtens = opts.dbExtens

outDir='/sps/lsst/users/gris/MetricOutput'
nodither = opts.nodither
splitSky = opts.splitSky

toprocess = np.genfromtxt(dbList,dtype=None,names=['dbName','simuType','nside','coadd','fieldType','nproc'])

print('there',toprocess)

n_per_slice = 1
n_process = len(toprocess)
lproc = list(range(0,n_process,n_per_slice))
if splitSky:
    RAs = np.linspace(0.,360.,11)
else:
    RAs = [0.,360.]

for i,val in enumerate(lproc):
#proc in toprocess:
    for ira in range(len(RAs)-1):
        RA_min = RAs[ira]
        RA_max = RAs[ira+1]
        batch_new(dbDir,dbExtens,'run_scripts/metrics/run_metrics_fromnpy',outDir,8,1,metricName,toprocess[val:val+n_per_slice],nodither,RA_min,RA_max)
   
                      
if (n_process & 1)&(n_per_slice>1):
    for ira in range(len(RAs)-1):
        RA_min = Ras[ira]
        RA_max = Ras[ira+1]
        batch_new(dbDir,dbExtens,'run_scripts/metrics/run_metrics_fromnpy',outDir,8,1,metricName,toprocess[-1],nodither,RA_min,RA_max)

