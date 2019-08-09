import matplotlib
#matplotlib.use('agg')
import numpy as np
import healpy as hp
from metricWrapper import CadenceMetricWrapper, SNRMetricWrapper
from metricWrapper import SNRRateMetricWrapper, NSNMetricWrapper
from metricWrapper import SLMetricWrapper
from sn_tools.sn_obs import renameFields, pixelate, season, GetShape, ObsPixel
from optparse import OptionParser
import time
import multiprocessing
import os
import pandas as pd

def selectDD(obs, nside):

    print(obs.dtype)

    print(np.unique(obs['proposalId']))

    propId = list(np.unique(obs['proposalId']))

    if len(propId) > 3:
        idx = obs['proposalId'] == 5
        return pixelate(obs[idx], nside, RaCol='fieldRA', DecCol='fieldDec')
    else:
        names = obs.dtype.names
        if 'fieldId' in names:
            print(np.unique(obs[['fieldId','note']]))
            idx = obs['fieldId'] == 0
            return pixelate(obs[idx], nside, RaCol='fieldRA', DecCol='fieldDec')
        else:
            """this is difficult
               we do not have other ways to identify
               DD except by selecting pixels with a large number of visits
            """
            pixels = pixelate(obs, nside, RaCol='fieldRA', DecCol='fieldDec')
            
            df = pd.DataFrame(np.copy(pixels))

            groups = df.groupby('healpixID').filter(lambda x: len(x)>5000)
            
            group_DD = groups.groupby(['fieldRA','fieldDec']).filter(lambda x: len(x)>4000)


            #return np.array(group_DD.to_records().view(type=np.matrix))
            return group_DD.to_records(index=False)

def loop(healpixels, obsFocalPlane, band, metricList, j=0, output_q=None):

    resfi = {}
    for metric in metricList:
        resfi[metric.name] = None

    for healpixID in healpixels:
        time_ref = time.time()
        obsMatch = obsFocalPlane.matchFast(healpixID)
        #print('after obs match',time.time()-time_ref)
        resdict = {}
        for metric in metricList:
            # print('metric',metric.name)
            # print(metric.run(band,seasonsel))
            # print('FP')
            # print(metric.run(band,season(obsMatch)))
            """
            if band != 'all':
                resdict[metric.name] = metric.run(band, season(obsMatch))
            else:
                if metric.name == 'CadenceMetric':
                    resdict[metric.name] = metric.run(band, season(obsMatch))
            """
            resdict[metric.name] = metric.run(season(obsMatch))
        for key in resfi.keys():
            if resdict[key] is not None:
                if resfi[key] is None:
                    resfi[key] = resdict[key]
                else:
                    #print('vstack',key,resfi[key],len(resdict[key]))
                    #resfi[key] = np.vstack([resfi[key], resdict[key]])
                    resfi[key] = np.concatenate((resfi[key], resdict[key]))

    if output_q is not None:
        return output_q.put({j: resfi})
    else:
        return resfi


parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched',
                  help="db name [%default]")
parser.add_option("--dbDir", type="str", default='', help="db dir [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="healpix nside [%default]")
parser.add_option("--band", type="str", default='r', help="band [%default]")
parser.add_option("--nproc", type="int", default='8',
                  help="number of proc  [%default]")
parser.add_option("--fieldtype", type="str", default='DD', help="field type DD or WFD[%default]")
parser.add_option("--x1", type="float", default='0.0',
                  help="SN x1 [%default]")
parser.add_option("--color", type="float", default='0.0',
                  help="SN color [%default]")
parser.add_option("--zmax", type="float", default='1.2',
                  help="zmax for simu [%default]")


opts, args = parser.parse_args()

print('Start processing...')


# Load file

dbDir = opts.dbDir
if dbDir == '':
    dbDir = '/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db'
dbName = opts.dbName

nside = opts.nside
band = opts.band
seasons = -1
nproc = opts.nproc
fieldtype = opts.fieldtype
x1 = opts.x1
color = opts.color
zmax = opts.zmax

pixArea = hp.nside2pixarea(nside, degrees=True)
outDir = 'MetricOutput'
if not os.path.isdir(outDir):
    os.makedirs(outDir)
# List of (instance of) metrics to process
metricList = []
"""
if band != 'all':
    metricList.append(SNRMetricWrapper(z=0.3))
metricList.append(CadenceMetricWrapper(season=seasons))
"""
#metricList.append(SNRRateMetricWrapper(z=0.3))
metricList.append(NSNMetricWrapper(fieldtype=fieldtype,pixArea=pixArea,season=-1))

#metricList.append(SLMetricWrapper(season=-1, nside=64))

# loading observations

observations = np.load('{}/{}.npy'.format(dbDir, dbName))

"""
if fieldtype =='WFD':
    idx = observations['proposalId'] == 3
    observations = observations[idx]
"""

observations = renameFields(observations)

# this is a "simple" tessalation using healpix
if fieldtype == 'DD':
    pixels = selectDD(observations,nside)
else:
    pixels = pixelate(observations, nside, RaCol='fieldRA', DecCol='fieldDec')

print('number of fields',len(np.unique(pixels['healpixID'])))
# this is a more complicated tessalation using a LSST Focal Plane
# Get the shape to identify overlapping obs
shape = GetShape(nside)
scanzone = shape.shape()
##
time_ref = time.time()
obsFocalPlane = ObsPixel(nside=nside, data=observations,
                         scanzone=scanzone, RaCol='fieldRA', DecCol='fieldDec')

print('obsfocal plane',time.time()-time_ref)

#healpixels = np.unique(obsPixels['healpixID'])[:10]
#res = loop(healpixels, obsFocalPlane, band, metricList)

# print(res)

timeref = time.time()

healpixels = np.unique(pixels['healpixID'])
npixels = int(len(healpixels))
delta = npixels
if nproc > 1:
    delta = int(delta/(nproc))

tabpix = range(0, npixels, delta)
if npixels not in tabpix:
    tabpix = np.append(tabpix, npixels)

tabpix = tabpix.tolist()


#if tabpix[-1]-tabpix[-2] <= 100:
#    tabpix.remove(tabpix[-2])

print(tabpix, len(tabpix))
result_queue = multiprocessing.Queue()
for j in range(len(tabpix)-1):
#for j in range(6,7):
    ida = tabpix[j]
    idb = tabpix[j+1]
    #print('go', ida, idb)
    p = multiprocessing.Process(name='Subprocess-'+str(j), target=loop, args=(
        healpixels[ida:idb], obsFocalPlane, band, metricList, j, result_queue))
    p.start()


resultdict = {}
for i in range(len(tabpix)-1):
    resultdict.update(result_queue.get())

for p in multiprocessing.active_children():
    p.join()

restot = {}
for metric in metricList:
    restot[metric.name] = None

for key, vals in resultdict.items():
    for keyb in vals.keys():
        if restot[keyb] is None:
            restot[keyb] = vals[keyb]
        else:
            restot[keyb] = np.concatenate((restot[keyb], vals[keyb]))

for key, vals in restot.items():
    np.save('{}/{}_{}.npy'.format(outDir, dbName, key), np.copy(vals))

print('Done', time.time()-timeref)
