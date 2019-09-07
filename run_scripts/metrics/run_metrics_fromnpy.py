import matplotlib.pyplot as plt
#matplotlib.use('agg')
import numpy as np
import healpy as hp
from metricWrapper import CadenceMetricWrapper, SNRMetricWrapper
from metricWrapper import SNRRateMetricWrapper, NSNMetricWrapper
from metricWrapper import SLMetricWrapper
from sn_tools.sn_obs import renameFields, pixelate, GetShape, ObsPixel
from sn_tools.sn_obs import season as seasoncalc
from optparse import OptionParser
import time
import multiprocessing
import os
import pandas as pd
from astropy_healpix import HEALPix
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy.lib.recfunctions as rf
from sn_stackers.coadd_stacker import CoaddStacker
from sn_tools.sn_obs import OverlapGnomonic

def getFields(obs, fieldIds):

    obs = None
    
    print('hhh',np.unique(observations['fieldId']))

    val = np.unique(observations[['fieldRA','fieldDec']])
    #plt.plot(val['fieldRA'],val['fieldDec'],'ko')
    idx = observations['fieldId']==0
    #sel = np.unique(observations[idx][['fieldRA','fieldDec']])
    sel = observations[idx]
    print(observations)
    #plt.plot(sel['fieldRA'],sel['fieldDec'],'b*')
    plt.plot(sel['observationStartMJD'],sel['fieldRA'],'ko')

    plt.show()
    



    print('selection',fieldIds)
    for fieldId in fieldIds:
        idf = observations['fieldId']==fieldId
        if obs is None:
            obs = observations[idf]
        else:
            obs = np.concatenate((obs,observations[idf]))
    return obs

def selectDD(obs, nside,fieldIds):

    print(obs.dtype)

    print(np.unique(obs['proposalId']))

    propId = list(np.unique(obs['proposalId']))

    if len(propId) > 3:
        #idx = obs['proposalId'] == 5
        obser = getFields(obs,fieldIds)
        return pixelate(obser, nside, RaCol='fieldRA', DecCol='fieldDec')
    else:
        names = obs.dtype.names
        if 'fieldId' in names:
            """
            print(np.unique(obs[['fieldId','note']]))
            """
            obser = getFields(obs,fieldIds)
            return pixelate(obser, nside, RaCol='fieldRA', DecCol='fieldDec')
        else:
            """this is difficult
               we do not have other ways to identify
               DD except by selecting pixels with a large number of visits
            """
            pixels = pixelate(obs, nside, RaCol='fieldRA', DecCol='fieldDec')
            
            
            df = pd.DataFrame(np.copy(pixels))

            print('ooo',np.unique(pixels['pixRA','pixDec']))
            groups = df.groupby('healpixID').filter(lambda x: len(x)>5000)
            
            group_DD = groups.groupby(['fieldRA','fieldDec']).filter(lambda x: len(x)>4000)


            #return np.array(group_DD.to_records().view(type=np.matrix))
            return group_DD.to_records(index=False)

def loop(healpixels,band, metricList, shape,observations,j=0, output_q=None):

    display = True
    resfi = {}
    for metric in metricList:
        resfi[metric.name] = None
    
    print('starting here')
    time_ref = time.time()
    print('rr',len(observations))
    observations = seasoncalc(observations)
    print('season',time.time()-time_ref,np.unique(observations['season']))
    idx = observations['season'] == 1
    observations = observations[idx]
    print('bbbb',len(observations))


    for pixel in healpixels:
    #for healpixID in [35465]:
        time_ref = time.time()
        ax = None
        if display:
            fig,ax = plt.subplots()
        
        #scanzone = shape.shape()
        #plt.show()
        ##
        time_ref = time.time()
        #scanzone = shape.shape(healpixID=healpixID)

        obsFocalPlane = ObsPixel(nside=nside, data=observations,
                                 RaCol='fieldRA', DecCol='fieldDec')

        
        obsMatch = obsFocalPlane.matchFast(pixel,ax=ax)
        #obsMatch = obsFocalPlane.matchQuery(healpixID)
        if display:
            fig.suptitle(pixel['healpixID'])
            plt.show()
        if obsMatch is None:
            continue
        print(len(obsMatch))
        #print(obsMatch.dtype)
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
            #print(season(obsMatch))
            #resdict[metric.name] = metric.run(season(obsMatch))
            resdict[metric.name] = metric.run(obsMatch)
        for key in resfi.keys():
            if resdict[key] is not None:
                if resfi[key] is None:
                    resfi[key] = resdict[key]
                else:
                    #print('vstack',key,resfi[key],resdict[key])
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
parser.add_option("--dithering", type="int", default='0',
                  help="dithering for DDF [%default]")
parser.add_option("--overlap", type="float", default='0.9',
                  help="overlap focal plane/pixel [%default]")


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
dithering = opts.dithering
overlap = opts.overlap

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
metricList.append(NSNMetricWrapper(fieldtype=fieldtype,pixArea=pixArea,season=-1,overlap=overlap,nside=nside))

#metricList.append(SLMetricWrapper(season=-1, nside=64))

# loading observations


observations = np.load('{}/{}.npy'.format(dbDir, dbName))
observations = renameFields(observations)
"""
print(observations.dtype)
stacker = CoaddStacker()

observations = stacker._run(observations)
"""

#print(obs.dtype)


"""
if fieldtype =='WFD':
    idx = observations['proposalId'] == 3
    observations = observations[idx]
"""


fieldIds = [290,744, 1427, 2412, 2786]
fieldIds = [0]
#fieldIds = [290]
#fieldIds = [290,1427, 2412, 2786]
# this is a "simple" tessalation using healpix
if fieldtype == 'DD':
    pixels = selectDD(observations,nside,fieldIds)
    observations = getFields(observations,fieldIds)
    print(np.unique(observations['fieldId']))
else:
    pixels = pixelate(observations, nside, RaCol='fieldRA', DecCol='fieldDec')

print('number of fields',len(np.unique(pixels['healpixID'])),len(pixels['healpixID']))

pixels = np.unique(pixels[['healpixID','pixRa','pixDec']])

print(pixels)

if dithering:
    hppix = HEALPix(nside=nside, order='nested')
    pixDither = None
    for (pixRa,pixDec) in pixels[['pixRa','pixDec']]:
        healpixID_around = hppix.cone_search_lonlat(pixRa * u.deg, pixDec* u.deg, radius=3*u.deg)
        coordpix = hp.pix2ang(nside, healpixID_around, nest=True, lonlat=True) 
        coords = SkyCoord(coordpix[0], coordpix[1], unit='deg')
        #print(coordpix[0])
        arr = np.array(healpixID_around,dtype=[('healpixID', 'i8')])
        arr = rf.append_fields(arr, 'pixRa', coordpix[0])
        arr = rf.append_fields(arr, 'pixDec', coordpix[1])
        print(pixRa,pixDec,len(arr))
        #arr = np.array([healpixID_around,coords[0],coords[1]])
        #print(arr)
        if pixDither is None:
            pixDither = arr
        else:
            pixDither = np.concatenate((pixDither,arr))
    pixels = pixDither
    print("number of pixels for dithering",len(pixels))
# this is a more complicated tessalation using a LSST Focal Plane
# Get the shape to identify overlapping obs
shape = GetShape(nside, overlap)
#fig, ax = plt.subplots()


#print('obsfocal plane',time.time()-time_ref)

#healpixels = np.unique(obsPixels['healpixID'])[:10]
#res = loop(healpixels, obsFocalPlane, band, metricList)

# print(res)

"""
overlapdisp = OverlapGnomonic(nside)

pointings = list(np.unique(observations[['fieldRA','fieldDec']]))


print('rrrr',pointings)

for pointing in pointings:
    print(pointing)
    fign, axn = plt.subplots()
    overlapdisp.overlap_pixlist(np.unique(pixels[['healpixID','pixRa','pixDec']]),pointing,ax=axn)
    axn.set_xlim(pointing[0]-6.,pointing[0]+6.)
    axn.set_ylim(pointing[1]-6.,pointing[1]+6.)

plt.show()
"""

timeref = time.time()

healpixels = np.unique(pixels[['healpixID','pixRa','pixDec']])
npixels = int(len(healpixels))
delta = npixels
if nproc > 1:
    delta = int(delta/(nproc))

tabpix = range(0, npixels, delta)
if npixels not in tabpix:
    tabpix = np.append(tabpix, npixels)

tabpix = tabpix.tolist()

if nproc > 1:
    if tabpix[-1]-tabpix[-2] <= 10:
        tabpix.remove(tabpix[-2])

print(tabpix, len(tabpix))
result_queue = multiprocessing.Queue()
for j in range(len(tabpix)-1):
#for j in range(4,5):
    ida = tabpix[j]
    idb = tabpix[j+1]
    #print('go', ida, idb)
    p = multiprocessing.Process(name='Subprocess-'+str(j), target=loop, args=(
        healpixels[ida:idb],band, metricList, shape,observations,j, result_queue))
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
