import matplotlib
matplotlib.use('agg')
import numpy as np
import healpy as hp
from metricWrapper import CadenceMetricWrapper, SNRMetricWrapper
from sn_tools.sn_obs import renameFields, pixelate, season, GetShape, ObsPixel
from optparse import OptionParser
import time
import multiprocessing

def loop(healpixels,obsFocalPlane,band,metricList,j=0, output_q=None):

    resfi = {}
    for metric in metricList:
        resfi[metric.name] = None

    for healpixID in healpixels:
        obsMatch = obsFocalPlane.matchFast(healpixID)
        resdict = {}
        for metric in metricList:
            #print('metric',metric.name)
            #print(metric.run(band,seasonsel))
            #print('FP')
            #print(metric.run(band,season(obsMatch)))
            resdict[metric.name] = metric.run(band,season(obsMatch))
    
        for key in resfi.keys():
            if resdict[key] is not None:
                if resfi[key] is None:
                    resfi[key] = resdict[key]
                else:
                    #print(key,resfi[key],resdict[key])
                    resfi[key] = np.concatenate((resfi[key],resdict[key]))

    if output_q is not None:
        return output_q.put({j:resfi})
    else:
        return resfi


parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched', help="db name [%default]")
parser.add_option("--dbDir", type="str", default='', help="db dir [%default]")
parser.add_option("--nside", type="int", default=64, help="healpix nside [%default]")
parser.add_option("--band", type="str", default='r', help="band [%default]")
parser.add_option("--nproc", type="int", default='8', help="number of proc  [%default]")

opts, args = parser.parse_args()

print('Start processing...')


#Load file

dbDir = opts.dbDir
if dbDir == '':
    dbDir = '/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db'
dbName = opts.dbName

nside = opts.nside
band = opts.band
seasons = -1
nproc = opts.nproc
outDir = '/sps/lsst/users/gris/MetricOutput'

# List of (instance of) metrics to process
metricList = []
metricList.append(SNRMetricWrapper(z=0.3))
metricList.append(CadenceMetricWrapper(season=seasons))

#loading observations

observations = np.load('{}/{}.npy'.format(dbDir, dbName))
observations = renameFields(observations)

# this is a "simple" tessalation using healpix
pixels = pixelate(observations, nside,RaCol='fieldRA',DecCol='fieldDec')

# this is a more complicated tessalation using a LSST Focal Plane
## Get the shape to identify overlapping obs
shape = GetShape(nside)
scanzone = shape.shape()
## 
obsFocalPlane = ObsPixel(nside=nside,data=observations,scanzone=scanzone, RaCol='fieldRA',DecCol='fieldDec')

#healpixels = np.unique(obsPixels['healpixID'])[:10]
#res = loop(healpixels, obsFocalPlane, band, metricList)
    
#print(res)

timeref = time.time()

healpixels = np.unique(pixels['healpixID'])
npixels = int(len(healpixels))
delta = npixels
if nproc > 1:
    delta = int(delta/(nproc))

tabpix = range(0,npixels,delta)
if npixels not in tabpix:
    tabpix = np.append(tabpix,npixels)

tabpix = tabpix.tolist()

if tabpix[-1]-tabpix[-2]<= 100:
    tabpix.remove(tabpix[-2])

print(tabpix,len(tabpix))
result_queue = multiprocessing.Queue()
for j in range(len(tabpix)-1):
#for j in range(5,6):
    ida = tabpix[j]
    idb = tabpix[j+1]
    p=multiprocessing.Process(name='Subprocess-'+str(j),target=loop,args=(healpixels[ida:idb],obsFocalPlane,band,metricList,j,result_queue))
    p.start()
    

resultdict = {}
for i in range(len(tabpix)-1):
    resultdict.update(result_queue.get())

for p in multiprocessing.active_children():
    p.join()

restot = {}
for metric in metricList:
    restot[metric.name] = None

for key,vals in resultdict.items():
    for keyb in vals.keys():
        if restot[keyb] is None:
            restot[keyb] = vals[keyb]
        else:
            restot[keyb] = np.concatenate((restot[keyb],vals[keyb]))

for key, vals in restot.items():
    np.save('{}/{}_{}_{}.npy'.format(outDir,dbName,key,band),np.copy(vals))

print('Done',time.time()-timeref)

