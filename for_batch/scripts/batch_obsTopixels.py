import os
import numpy as np
from optparse import OptionParser
import pandas as pd
from sn_tools.sn_batchutils import BatchIt

"""
def process(dbDir, dbName, dbExtens, scriptref, outDir, nproc,
          saveData, fieldType, simuType, nside, fieldNames, nclusters,
            RAmin,RAmax,nRA,Decmin,Decmax,nDec,radius,mode='batch'):

    print('boa',RAmin,RAmax,nRA,Decmin,Decmax,nDec,radius,mode)

    cwd = os.getcwd()
    dirScript = cwd + "/scripts"

    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)

    # dbName = dbName.decode()
    # fieldType = fieldType.decode()

    id = '{}_{}_{}'.format(
        dbName, nside, fieldType)

    name_id = 'obsTopixels_{}'.format(id)
    logName = dirLog + '/'+name_id+'.log'
    errlogName = dirLog + '/'+name_id+'.err'


    dict_batch = {}
    dict_batch['--account'] = 'lsst'
    dict_batch['-L'] = 'sps'
    dict_batch['--time'] = '10:00:00'
    dict_batch['--mem'] = '20G'
    dict_batch['--output'] = logName
    #dict_batch['--cpus-per-task'] = str(nproc)
    dict_batch['-n'] = 8
    dict_batch['--error'] = errlogName

    # fill the script
    scriptName = dirScript+'/'+name_id+'.sh'
    script = open(scriptName, "w")
    
    #script.write(qsub + "\n")
    script.write("#!/bin/env bash\n") 
    for key, vals in dict_batch.items():
         script.write("#SBATCH {} {} \n".format(key,vals))

    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh Linux -5\n")
    script.write("echo 'sourcing done' \n")

    if fieldType == 'WFD':
        cmd = cmdb(dbDir, dbName, dbExtens, scriptref, outDir, nproc,
                   saveData, fieldType, simuType, nside,'',0,
                   RAmin,RAmax,nRA,Decmin,Decmax,nDec)
        script.write(cmd + " \n")
        if mode == 'interactive':
            os.system(cmd)
    if fieldType == 'DD':
        for fieldName in fieldNames:
            cmd = cmdb(dbDir, dbName, dbExtens, scriptref, outDir, nproc,
                       saveData, fieldType, simuType, nside, fieldName, nclusters,
                       RAmin,RAmax,nRA,Decmin,Decmax,nDec,radius)
            script.write(cmd + " \n")
            if mode == 'interactive':
                os.system(cmd)
    script.write("EOF" + "\n")
    script.close()
    if mode == 'batch':
        os.system("sbatch "+scriptName)


def cmdb(dbDir, dbName, dbExtens, scriptref, outDir, nproc,
         saveData, fieldType, simuType, nside, fieldName='', nclusters=0,
         RAmin=0.,RAmax=360.,nRA=10,Decmin=-80.,Decmax=20.,nDec=4,radius=4.):

    cmd = 'python {}'.format(scriptref)
    cmd += ' --dbDir {}'.format(dbDir)
    cmd += ' --dbExtens {}'.format(dbExtens)
    cmd += ' --dbName {}'.format(dbName)
    cmd += ' --nproc {}'.format(nproc)
    cmd += ' --nside {}'.format(nside)
    #cmd += ' --simuType {}'.format(simuType)
    cmd += ' --saveData {}'.format(saveData)
    cmd += ' --outDir {}'.format(outDir)
    cmd += ' --fieldType {}'.format(fieldType) 
    cmd += ' --RAmin {}'.format(RAmin)
    cmd += ' --RAmax {}'.format(RAmax)
    cmd += ' --nRA {}'.format(nRA)
    cmd += ' --Decmin {}'.format(Decmin)
    cmd += ' --Decmax {}'.format(Decmax)
    cmd += ' --nDec {}'.format(nDec)
    cmd += ' --radius {}'.format(radius)

    if fieldName != '':
        cmd += ' --fieldName {}'.format(fieldName)
        cmd += ' --nclusters {}'.format(nclusters)

    return cmd
"""

def bbatch(scriptref,params,vvars,val,bid):
    
    processName =  'obsTopixels_{}'.format(bid)
    mybatch = BatchIt(processName=processName)
    for vv in vvars:
        params[vv] = val[vv]
    mybatch.add_batch(scriptref,params)
    mybatch.go_batch()

def make_scr(scriptref,params,scrName):
    script = open(scrName, "w")

    script.write('#!/bin/bash \n')
    cmd = 'python {}'.format(scriptref)
    
    for key,vals in params.items():
        cmd += ' --{} {}'.format(key,vals)

    script.write(cmd+'\n')


    cmd = "srun -p htc_interactive --ntasks-per-core 8 {}".format(scrName)

    print('executing',cmd)
    os.system(cmd)


parser = OptionParser()

parser.add_option("--dbList", type="str", default='input/obsTopixels/List_Db_DD.csv',
                  help="dbList to process  [%default]")
parser.add_option("--nodither", type="str", default='',
                  help="to remove dithering [%default]")
parser.add_option("--outDir", type="str",
                  default='/sps/lsst/users/gris/ObsPixelized',
                  help="output directory[%default]")
parser.add_option("--nclusters", type=int,
                  default=6,
                  help="number of clusters - for DD only[%default]")
parser.add_option("--RAmin", type=float, default=0.,
                  help="RA min for obs area - [%default]")
parser.add_option("--RAmax", type=float, default=360.,
                  help="RA max for obs area - [%default]")
parser.add_option("--nRA", type=int, default=10,
                  help="number of RA patches - [%default]")
parser.add_option("--Decmin", type=float, default=-1.0,
                  help="Dec min for obs area - [%default]")
parser.add_option("--Decmax", type=float, default=-1.0,
                  help="Dec max for obs area - [%default]")
parser.add_option("--nDec", type=int, default=1,
                  help="number of Dec patches - [%default]")
parser.add_option("--radius", type=float, default=4.,
                  help="radius for pixel arounf center [DD only] - [%default]")
parser.add_option("--DDFs", type=str, default='COSMOS,XMM-LSS,ELAIS,CDFS,ADFS1',
                  help="list of DDF ro consider - [%default]")
parser.add_option("--mode", type=str, default='batch',
                  help="mode to run (batch/interactive) - [%default]")

opts, args = parser.parse_args()

print('Start processing...', opts)

dbList = opts.dbList
outDir = opts.outDir
RAmin = opts.RAmin
RAmax = opts.RAmax
Decmin = opts.Decmin
Decmax = opts.Decmax
nRA = opts.nRA
nDec = opts.nDec
radius = opts.radius
mode = opts.mode
nclusters = opts.nclusters

"""
toprocess = np.genfromtxt(dbList, dtype=None, names=[
                          'dbName', 'simuType', 'nside', 'coadd', 'fieldType', 'nproc'])
"""
toprocess = pd.read_csv(dbList, comment='#')

# print('there', toprocess)
scriptref = 'run_scripts/obs_pixelize/run_obs_to_pixels.py'
fieldNames = []
#fieldDD = ['COSMOS', 'CDFS', 'ELAIS', 'XMM-LSS', 'ADFS1', 'ADFS2']
fieldDD = opts.DDFs.split(',')

params = {}
params['outDir'] = outDir
params['RAmin'] = RAmin
params['RAmax'] = RAmax
params['nRA'] = nRA
params['Decmin'] = Decmin
params['Decmax'] = Decmax
params['nDec'] = nDec
params['radius'] = radius
params['saveData'] = 1

vvars = ['dbDir', 'dbName', 'dbExtens','nproc','fieldType','simuType','nside']

for io, val in toprocess.iterrows():
    bid = '{}_{}_{}'.format(val['dbName'], val['nside'], val['fieldType'])
    if mode == 'batch':
        if val['fieldType'] == 'DD':
            fieldNames = fieldDD
            params['nclusters'] = nclusters
            for fieldName in fieldNames:
                bbid = '{}_{}'.format(bid,fneldName)
                bbatch(scriptref,params,vvars+['fieldName'],val,bbid)
        else:
            bbatch(scriptref,params,vvars,val,bid)
    else:
        scrName = 'test_{}.sh'.format(bid)
        for vv in vvars:
             params[vv] = val[vv]
        make_scr(scriptref,params,scrName)
        os.system('chmod +x {}'.format(scrName))


    """
    process(val['dbDir'], val['dbName'], val['dbExtens'], scriptref, outDir, val['nproc'],
              1, val['fieldType'], val['simuType'], val['nside'], fieldNames, opts.nclusters,
              RAmin,RAmax,nRA,Decmin,Decmax,nDec,radius,mode)
    """ 
        
    
    
