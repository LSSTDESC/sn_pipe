import os
from optparse import OptionParser

parser = OptionParser(description='perform cosmo fit')

parser.add_option("--dbDir", type="str",default='/sps/lsst/users/gris/Fake_Observations',help="fake obs directory [%default]")
parser.add_option("--outDir", type="str",default='/sps/lsst/users/gris/Fakes/Simu',help="output directory [%default]")
parser.add_option("--dbNames", type=str, default="DD_0.65,DD_0.70,DD_0.75,DD_0.80,DD_0.85,DD_0.90",help="configurations to process [%default]")

parser.add_option("--snTypes", type=str, default="faintSN,allSN",help="type of SN to simulate [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
outDir = opts.outDir
dbNames = opts.dbNames.split(',')
snTypes = opts.snTypes

script = 'python for_batch/scripts/batch_fake_simu.py'

for dbName in dbNames:
    w_ = script
    w_ += ' --dbName {}'.format(dbName)
    w_ += ' --dbDir {}'.format(dbDir)
    w_ += ' --outDir {}'.format(outDir)
    w_ += ' --snTypes {}'.format(snTypes)
    print(w_)
    os.system(w_)
