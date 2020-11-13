import os
from optparse import OptionParser

def simulation(fieldName,dbName,dbDir,dbExtens,outDir,mode):
     
    cmd = 'python for_batch/scripts/batch_dd_simu.py'
    cmd += ' --fieldName {}'.format(fieldName)
    cmd += ' --dbName {}'.format(dbName)
    cmd += ' --dbDir {}'.format(dbDir)
    cmd += ' --dbExtens {}'.format(dbExtens)
    cmd += ' --outDir {}'.format(outDir)
    cmd += ' --mode {}'.format(mode)
    print(cmd)
    os.system(cmd)

def fit(fieldName,dbName,simuDir,outDir,mode,snrmin):

    cmd = 'python for_batch/scripts/batch_dd_fit.py'
    cmd += ' --fieldName {}'.format(fieldName)
    cmd += ' --dbName {}'.format(dbName)
    cmd += ' --simuDir {}'.format(simuDir)
    cmd += ' --outDir {}'.format(outDir)
    cmd += ' --mode {}'.format(mode)
    cmd += ' --snrmin {}'.format(snrmin)
    print(cmd)
    os.system(cmd)

parser = OptionParser()

parser.add_option("--dbName", type="str", default='descddf_v1.5_10yrs',help="dbName to process  [%default]")
parser.add_option("--dbDir", type="str", default='/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.5/npy',help="dbDir to process  [%default]")
parser.add_option("--dbExtens", type="str", default='npy',help="dbDir extens [%default]")
parser.add_option("--simuDir", type="str", default='/sps/lsst/users/gris/DD/Simu',help="simu dir [%default]")
parser.add_option("--fitDir", type="str", default='/sps/lsst/users/gris/DD/Fit',help="output directory [%default]")
parser.add_option("--action", type="str", default='simulation',help="what to do: simulation or fit [%default]")
parser.add_option("--mode", type="str", default='batch',help="running mode batch/interactive [%default]")
parser.add_option("--snrmin", type=float, default=1.,help="min snr for LC point fit[%default]")

opts, args = parser.parse_args()

dbName = opts.dbName
dbDir = opts.dbDir
dbExtens = opts.dbExtens
simuDir = opts.simuDir
fitDir = opts.fitDir
mode = opts.mode
snrmin = opts.snrmin

DDF = ['COSMOS','CDFS','ELAIS','XMM-LSS','ADFS1','ADFS2']
#DDF = ['ELAIS']

for dd in DDF:
    if opts.action == 'simulation':
        simulation(dd,dbName,dbDir,dbExtens,simuDir,mode)
    if opts.action == 'fit':
        fit(dd,dbName,simuDir,fitDir,mode,snrmin)
