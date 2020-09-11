import os
import numpy as np
from optparse import OptionParser
import pandas as pd

def batch(dbDir, dbName,dbExtens, scriptref, outDir, nproc,
          saveData,fieldType,simuType,nside):

    cwd = os.getcwd()
    dirScript = cwd + "/scripts"

    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)

    #dbName = dbName.decode()
    #fieldType = fieldType.decode()

    id = '{}_{}_{}'.format(
        dbName, nside, fieldType)

    name_id = 'obsTopixels_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'

   
    qsub = 'qsub -P P_lsst -l sps=1,ct=5:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
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
    
    cmd = 'python {}'.format(scriptref)
    cmd += ' --dbDir {}'.format(dbDir)
    cmd += ' --dbExtens {}'.format(dbExtens)
    cmd += ' --dbName {}'.format(dbName)
    cmd += ' --nproc {}'.format(nproc)
    cmd += ' --nside {}'.format(nside)
    cmd += ' --simuType {}'.format(simuType)
    cmd += ' --saveData {}'.format(saveData)
    cmd += ' --outDir {}'.format(outDir)
    cmd += ' --fieldType {}'.format(fieldType)

    script.write(cmd +" \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)



parser = OptionParser()

parser.add_option("--dbList", type="str", default='WFD.txt',
                  help="dbList to process  [%default]")
parser.add_option("--dbDir", type="str", 
                  default='/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.4/db', help="db dir [%default]")
parser.add_option("--dbExtens", type="str", default='db',
                  help="db extension [%default]")
parser.add_option("--nodither", type="str", default='',
                  help="to remove dithering [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="healpix nside[%default]")
parser.add_option("--outDir", type="str", 
                  default='/sps/lsst/users/gris/ObsPixelized',
                  help="output directory[%default]")

opts, args = parser.parse_args()

print('Start processing...',opts)

dbList = opts.dbList
dbDir = opts.dbDir
dbExtens = opts.dbExtens
outDir = opts.outDir

"""
toprocess = np.genfromtxt(dbList, dtype=None, names=[
                          'dbName', 'simuType', 'nside', 'coadd', 'fieldType', 'nproc'])
"""
toprocess = pd.read_csv(dbList, comment='#')

#print('there', toprocess)
scriptref='run_scripts/obs_pixelize/run_obs_to_pixels.py'
for io,val in toprocess.iterrows():
    batch(dbDir, val['dbName'],dbExtens,scriptref, outDir,val['nproc'],
          1,val['fieldType'],val['simuType'],opts.nside)
