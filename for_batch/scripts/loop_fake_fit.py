import os
from optparse import OptionParser

parser = OptionParser(description='perform cosmo fit')

parser.add_option("--simuDir", type="str",default='/sps/lsst/users/gris/Fakes/Simu',help="simulation directory [%default]")
parser.add_option("--outDir", type="str",default='/sps/lsst/users/gris/Fakes/Fit',help="output directory [%default]")
parser.add_option("--dbNames", type=str, default="DD_0.65,DD_0.70,DD_0.75,DD_0.80,DD_0.85,DD_0.90",help="configurations to process [%default]")

parser.add_option("--snTypes", type=str, default="faintSN,allSN",help="type of SN to simulate [%default]")

opts, args = parser.parse_args()

simuDir = opts.simuDir
outDir = opts.outDir
dbNames = opts.dbNames.split(',')
snTypes = opts.snTypes

script = 'python for_batch/scripts/batch_fake_fit.py'

for dbName in dbNames:
    w_ = script
    w_ += ' --dbName {}'.format(dbName)
    w_ += ' --simuDir {}'.format(simuDir)
    w_ += ' --outDir {}'.format(outDir)
    w_ += ' --snTypes {}'.format(snTypes)
    print(w_)
    #os.system(w_)
