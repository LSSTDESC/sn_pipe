import os
from optparse import OptionParser

parser = OptionParser(description='perform cosmo fit')

parser.add_option("--dbDir", type="str",default='/sps/lsst/users/gris/Fake_Observations',help="fake obs directory [%default]")
parser.add_option("--outDir", type="str",default='/sps/lsst/users/gris/Fakes/Simu',help="output directory [%default]")
parser.add_option("--dbNames", type=str, default="DD_0.65,DD_0.70,DD_0.75,DD_0.80,DD_0.85,DD_0.90",help="configurations to process [%default]")
parser.add_option("--x1sigma", type=int, default=0,help="shift of x1 parameter distribution[%default]")
parser.add_option("--colorsigma", type=int, default=0,help="shift of color parameter distribution[%default]")

parser.add_option("--snTypes", type=str, default="faintSN,allSN",help="type of SN to simulate [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
outDir = opts.outDir
dbNames = opts.dbNames.split(',')
snTypes = opts.snTypes
x1sigma = opts.x1sigma
colorsigma = opts.colorsigma

script = 'python for_batch/scripts/batch_fake_simu.py'

nabs_ref = dict(zip(['faintSN','allSN'],[-1,1500]))
nsnfactor_ref = dict(zip(['faintSN','allSN'],[100,100]))

sntest = snTypes.split(',')
print('aooo',sntest,nabs_ref)
nabs = ','.join(['{}'.format(nabs_ref[kk]) for kk in sntest])
nsnfactor = ','.join(['{}'.format(nsnfactor_ref[kk]) for kk in sntest])


for dbName in dbNames:
    w_ = script
    w_ += ' --dbName {}'.format(dbName)
    w_ += ' --dbDir {}'.format(dbDir)
    w_ += ' --outDir {}'.format(outDir)
    w_ += ' --snTypes {}'.format(snTypes)
    w_ += ' --x1sigma {}'.format(x1sigma)
    w_ += ' --colorsigma {}'.format(colorsigma)
    w_ += ' --nabs {}'.format(nabs)
    w_ += ' --nsnfactor {}'.format(nsnfactor)
    print(w_)
    os.system(w_)
