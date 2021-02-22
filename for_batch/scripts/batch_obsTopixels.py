import os
import numpy as np
from optparse import OptionParser
import pandas as pd


def batch(dbDir, dbName, dbExtens, scriptref, outDir, nproc,
          saveData, fieldType, simuType, nside, fieldNames, nclusters,
          RAmin,RAmax,nRA,Decmin,Decmax,nDec):

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
    log = dirLog + '/'+name_id+'.log'

    qsub = 'qsub -P P_lsst -l sps=1,ct=9:00:00,h_vmem=20G -j y -o {} -pe multicores {} <<EOF'.format(
        log, nproc)

    scriptName = dirScript+'/'+name_id+'.sh'

    script = open(scriptName, "w")
    script.write(qsub + "\n")
    script = open(scriptName, "w")
    script.write(qsub + "\n")
    script.write("#!/bin/env bash\n")
    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh Linux\n")
    script.write("echo 'sourcing done' \n")

    if fieldType == 'WFD':
        cmd = cmdb(dbDir, dbName, dbExtens, scriptref, outDir, nproc,
                   saveData, fieldType, simuType, nside,
                   RAmin,RAmax,nRA,Decmin,Decmax,nDec)
        script.write(cmd + " \n")
    if fieldType == 'DD':
        for fieldName in fieldNames:
            cmd = cmdb(dbDir, dbName, dbExtens, scriptref, outDir, nproc,
                       saveData, fieldType, simuType, nside, fieldName, nclusters,
            RAmin,RAmax,nRA,Decmin,Decmax,nDec)
            script.write(cmd + " \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)


def cmdb(dbDir, dbName, dbExtens, scriptref, outDir, nproc,
         saveData, fieldType, simuType, nside, fieldName='', nclusters=0,
         RAmin=0.,RAmax=360.,nRA=10,Decmin=-80.,Decmax=20.,nDec=4):

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

    if fieldName != '':
        cmd += ' --fieldName {}'.format(fieldName)
        cmd += ' --nclusters {}'.format(nclusters)

    return cmd


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
parser.add_option("--Decmin", type=float, default=-80.,
                  help="Dec min for obs area - [%default]")
parser.add_option("--Decmax", type=float, default=20.,
                  help="Dec max for obs area - [%default]")
parser.add_option("--nDec", type=int, default=4,
                  help="number of Dec patches - [%default]")

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

"""
toprocess = np.genfromtxt(dbList, dtype=None, names=[
                          'dbName', 'simuType', 'nside', 'coadd', 'fieldType', 'nproc'])
"""
toprocess = pd.read_csv(dbList, comment='#')

# print('there', toprocess)
scriptref = 'run_scripts/obs_pixelize/run_obs_to_pixels.py'
fieldNames = []
fieldDD = ['COSMOS', 'CDFS', 'ELAIS', 'XMM-LSS', 'ADFS1', 'ADFS2']
for io, val in toprocess.iterrows():
    if val['fieldType'] == 'DD':
        fieldNames = fieldDD
    batch(val['dbDir'], val['dbName'], val['dbExtens'], scriptref, outDir, val['nproc'],
          1, val['fieldType'], val['simuType'], val['nside'], fieldNames, opts.nclusters,
          RAmin,RAmax,nRA,Decmin,Decmax,nDec)
