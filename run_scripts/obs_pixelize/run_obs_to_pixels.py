import numpy as np
from optparse import OptionParser
import time
import multiprocessing
import os
from sn_tools.sn_obs import DataToPixels, ProcessPixels,renameFields,patchObs
from sn_tools.sn_io import getObservations
import pandas as pd

def processPatch(pointings,observations, nside, outDir, dbName, nodither=False, RACol='', DecCol='', j=0, output_q=None):

    print('processing area', j, pointings)

    time_ref = time.time()
    ipoint = 1
 
    datapixels = DataToPixels(nside, RACol, DecCol, j, outDir, dbName)
    pixelsTot = pd.DataFrame()
    for index, pointing in pointings.iterrows():
        ipoint += 1
        print('pointing',ipoint)

        # get the pixels
        pixels= datapixels(observations, pointing['RA'], pointing['Dec'],
                            pointing['radius_RA'], pointing['radius_Dec'], ipoint, nodither, display=False)

        # select pixels that are inside the original area
        
        idx = (pixels['pixRA']-pointing['RA'])>=-pointing['radius_RA']/2.
        idx &= (pixels['pixRA']-pointing['RA'])<pointing['radius_RA']/2.
        idx &= (pixels['pixDec']-pointing['Dec'])>=-pointing['radius_Dec']/2.
        idx &= (pixels['pixDec']-pointing['Dec'])<pointing['radius_Dec']/2.

        pixelsTot=pd.concat((pixelsTot,pixels[idx]),sort=False)
        
        #datapixels.plot(pixels)
    print('end of processing for', j, time.time()-time_ref)
    if output_q is not None:
        return output_q.put({j: pixelsTot})
    else:
        return pixelsTot
        
   

parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched',
                  help="db name [%default]")
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
parser.add_option("--dbDir", type="str",
                  default='/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db', help="db dir [%default]")
parser.add_option("--outDir", type="str", default='ObsPixelized',
                  help="output dir [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="healpix nside [%default]")
parser.add_option("--nproc", type="int", default='8',
                  help="number of proc  [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type DD or WFD[%default]")
parser.add_option("--remove_dithering", type="int", default='0',
                  help="remove dithering for DDF [%default]")
parser.add_option("--simuType", type="int", default='0',
                  help="flag for new simulations [%default]")
parser.add_option("--saveData", type="int", default='0',
                  help="flag to dump data on disk [%default]")
parser.add_option("--RAmin", type=float, default=0.,
                  help="RA min for obs area - for WDF only[%default]")
parser.add_option("--RAmax", type=float, default=360.,
                  help="RA max for obs area - for WDF only[%default]")
parser.add_option("--Decmin", type=float, default=-1.,
                  help="Dec min for obs area - for WDF only[%default]")
parser.add_option("--Decmax", type=float, default=-1.,
                  help="Dec max for obs area - for WDF only[%default]")
parser.add_option("--verbose", type=int, default=0,
                  help="verbose mode for the metric[%default]")

opts, args = parser.parse_args()

print('Start processing...', opts)

# prepare outputDir
nodither = ''
if opts.remove_dithering:
    nodither = '_nodither'
outputDir = '{}/{}{}'.format(opts.outDir,
                                opts.dbName, nodither)
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

# loading observations

observations = getObservations(opts.dbDir, opts.dbName, opts.dbExtens)

# rename fields

observations = renameFields(observations)

RACol = 'fieldRA'
DecCol = 'fieldDec'

if 'RA' in observations.dtype.names:
    RACol = 'RA'
    DecCol = 'Dec'
    
observations, patches = patchObs(observations, opts.fieldType,
                                 opts.dbName,
                                 opts.nside,
                                 opts.RAmin,opts.RAmax,
                                 opts.Decmin,opts.Decmax,
                                 RACol, DecCol,
                                 display=False)

print('observations', len(observations), len(patches))

timeref = time.time()

healpixels = patches
npixels = int(len(healpixels))


tabpix = np.linspace(0, npixels, opts.nproc+1, dtype='int')
print(tabpix, len(tabpix))
result_queue = multiprocessing.Queue()

# multiprocessing
for j in range(len(tabpix)-1):
    ida = tabpix[j]
    idb = tabpix[j+1]
    
    print('Field', healpixels[ida:idb])
    
    field = healpixels[ida:idb]
    
    p = multiprocessing.Process(name='Subprocess-'+str(j), target=processPatch, args=(
        healpixels[ida:idb],observations, opts.nside,
        outputDir, opts.dbName, opts.remove_dithering, RACol, DecCol, j, result_queue))
    p.start()



resultdict = {}

for i in range(opts.nproc):
    resultdict.update(result_queue.get())

for p in multiprocessing.active_children():
    p.join()

restot = pd.DataFrame()

# gather the results
for key, vals in resultdict.items():
    restot = pd.concat((restot,vals),sort=False)
    
#now grab 
if opts.saveData:
    outName = '{}/{}_{}_nside_{}_{}_{}_{}_{}.npy'.format(outputDir,
                                                      opts.dbName,opts.fieldType,opts.nside,opts.RAmin,opts.RAmax,opts.Decmin,opts.Decmax)
    np.save(outName,restot.to_records(index=False))
