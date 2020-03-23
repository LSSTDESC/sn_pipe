import numpy as np
import glob
from optparse import OptionParser


parser = OptionParser()

parser.add_option("--dbList", type="str", default='WFD.txt',
                  help="dbList to process  [%default]")
parser.add_option("--dbDir", type="str", 
                  default='/sps/lsst/users/gris/ObsPixelized', help="db dir [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="healpix nside[%default]")
parser.add_option("--outName", type="str",
                  default='ObsPixels_fbs14', help="db dir [%default]")

opts, args = parser.parse_args()

print('Start processing...',opts)

dbList = opts.dbList
dbDir = opts.dbDir
nside = opts.nside
outName = opts.outName

dbDir = '/sps/lsst/users/gris/ObsPixelized'

toprocess = np.genfromtxt(dbList, dtype=None, names=[
    'dbName', 'simuType', 'nside', 'coadd', 'fieldType', 'nproc'])


r = []
for dbName in toprocess['dbName']:
    dbName = dbName.decode()
    search_path ='{}/{}/*_nside_{}_*.npy'.format(dbDir,dbName,nside)
    print(search_path)
    files = glob.glob(search_path)
    npixels = 0
    for fi in files:
        tab = np.load(fi)
        npixels += len(np.unique(tab['healpixID']))

    print(dbName,npixels)
    r.append((dbName,npixels))


res = np.rec.fromrecords(r, names=['dbName','npixels'])

np.save('{}_nside_{}.npy'.format(outName,nside),res)
