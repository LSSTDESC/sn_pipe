import matplotlib.pyplot as plt
# matplotlib.use('agg')
import numpy as np
import healpy as hp
from metricWrapper import CadenceMetricWrapper, SNRMetricWrapper
from metricWrapper import SNRRateMetricWrapper, NSNMetricWrapper
from metricWrapper import SLMetricWrapper
from sn_tools.sn_obs import renameFields, pixelate, GetShape, ObsPixel, ProcessArea
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
from sn_tools.sn_obs import pavingSky
import glob
from numpy import genfromtxt

def getFields(observations, fieldIds, simuType=1):

    obs = None
    print(np.unique(observations['proposalId']))
    propId = list(np.unique(observations['proposalId']))

    # this is for the WFD
    if fieldIds == [1]:
        if len(propId) > 3:
            idx = observations['proposalId'] == 3
            return np.copy(observations[idx])
        elif  len(propId) ==2:
             idx = observations['proposalId'] == 1
             if simuType == 2:
                 idx = observations['proposalId'] == 0
             return np.copy(observations[idx])
        else:
            idx = observations['proposalId'] == 0
            return np.copy(observations[idx]) 

    """
    print('hhh',np.unique(observations['fieldId']))

    val = np.unique(observations[['fieldRa','fieldDec']])
    # plt.plot(val['fieldRA'],val['fieldDec'],'ko')
    idx = observations['fieldId']==0
    # sel = np.unique(observations[idx][['fieldRA','fieldDec']])
    sel = observations[idx]
    print(observations)
    # plt.plot(sel['fieldRA'],sel['fieldDec'],'b*')
    plt.plot(sel['observationStartMJD'],sel['fieldRa'],'ko')

    plt.show()
    """

    # this is for DDF
    print('selection', fieldIds,observations.dtype)
    for fieldId in fieldIds:
        idf = observations['fieldId'] == fieldId
        if obs is None:
            obs = observations[idf]
        else:
            obs = np.concatenate((obs, observations[idf]))
    return obs


def selectDD(obs, nside, fieldIds):

    print(obs.dtype)

    print(np.unique(obs['proposalId']))

    propId = list(np.unique(obs['proposalId']))

    if len(propId) > 3:
        # idx = obs['proposalId'] == 5
        obser = getFields(obs, fieldIds)
        return pixelate(obser, nside, RaCol='fieldRA', DecCol='fieldDec')
    else:
        names = obs.dtype.names
        if 'fieldId' in names:
            """
            print(np.unique(obs[['fieldId','note']]))
            """
            obser = getFields(obs, fieldIds)
            return pixelate(obser, nside, RaCol='fieldRA', DecCol='fieldDec')
        else:
            """this is difficult
               we do not have other ways to identify
               DD except by selecting pixels with a large number of visits
            """
            pixels = pixelate(obs, nside, RaCol='fieldRA', DecCol='fieldDec')

            df = pd.DataFrame(np.copy(pixels))

            groups = df.groupby('healpixID').filter(lambda x: len(x) > 5000)

            group_DD = groups.groupby(['fieldRA', 'fieldDec']).filter(
                lambda x: len(x) > 4000)

            # return np.array(group_DD.to_records().view(type=np.matrix))
            return group_DD.to_records(index=False)


def loop_area(pointings, band, metricList, observations, nside, outDir,dbName,saveData,nodither,RaCol,DecCol,j=0, output_q=None):

    resfi = {}
    #print(np.unique(observations[['fieldRA', 'fieldDec']]))
    
    print('processing pointings',j,pointings)
    
    for metric in metricList:
        resfi[metric.name] = None
        listf = glob.glob('{}/*_{}_{}*'.format(outDir,metric.name,j))
        if len(listf) > 0:
            for val in listf:
                os.system('rm {}'.format(val))

    #print(test)
    time_ref = time.time()
    #print('Starting processing', len(pointings),j)
    ipoint = 1
    #myprocess = ProcessArea(nside,'fieldRA', 'fieldDec',j,outDir,dbName,saveData)
    print('hhh',RaCol,DecCol)
    myprocess = ProcessArea(nside,RaCol,DecCol,j,outDir,dbName,saveData)
    for pointing in pointings:
        ipoint += 1
        #print('pointing',ipoint)
        
       
        """
        myprocess = ProcessArea(
            nside, pointing['Ra'], pointing['Dec'], pointing['radius'], pointing['radius'], 'fieldRA', 'fieldDec',j,outDir,dbName)
        """
        #resdict = myprocess.process(observations, metricList,ipoint)
        #print('obs',len(observations))
        resdict = myprocess(observations, metricList, pointing['Ra'], pointing['Dec'], pointing['radius'], pointing['radius'],ipoint,nodither,display=False)
        

    """
        for key in resfi.keys():
            if resdict[key] is not None:
                if resfi[key] is None:
                    resfi[key] = resdict[key]
                else:
                    # print('vstack',key,resfi[key],resdict[key])
                    # resfi[key] = np.vstack([resfi[key], resdict[key]])
                    resfi[key] = np.concatenate((resfi[key], resdict[key]))
    """
    print('end of processing for', j,time.time()-time_ref)

    """
    if output_q is not None:
        return output_q.put({j: resfi})
    else:
        return resfi
    """

def loop(healpixels, band, metricList, shape, observations, j=0, output_q=None):

    display = True
    resfi = {}
    for metric in metricList:
        resfi[metric.name] = None

    print('starting here')
    time_ref = time.time()
    print('rr', len(observations))
    observations = seasoncalc(observations)
    print('season', time.time()-time_ref, np.unique(observations['season']))
    idx = observations['season'] == 1
    observations = observations[idx]
    print('bbbb', len(observations))

    for pixel in healpixels:
        # for healpixID in [35465]:
        time_ref = time.time()
        ax = None
        if display:
            fig, ax = plt.subplots()

        # scanzone = shape.shape()
        # plt.show()
        ##
        time_ref = time.time()
        # scanzone = shape.shape(healpixID=healpixID)

        obsFocalPlane = ObsPixel(nside=nside, data=observations,
                                 RaCol='fieldRA', DecCol='fieldDec')

        obsMatch = obsFocalPlane.matchFast(pixel, ax=ax)
        # obsMatch = obsFocalPlane.matchQuery(healpixID)
        if display:
            fig.suptitle(pixel['healpixID'])
            plt.show()
        if obsMatch is None:
            continue
        print(len(obsMatch))
        # print(obsMatch.dtype)
        # print('after obs match',time.time()-time_ref)
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
            # print(season(obsMatch))
            # resdict[metric.name] = metric.run(season(obsMatch))
            resdict[metric.name] = metric.run(obsMatch)
        for key in resfi.keys():
            if resdict[key] is not None:
                if resfi[key] is None:
                    resfi[key] = resdict[key]
                else:
                    # print('vstack',key,resfi[key],resdict[key])
                    # resfi[key] = np.vstack([resfi[key], resdict[key]])
                    resfi[key] = np.concatenate((resfi[key], resdict[key]))

    if output_q is not None:
        return output_q.put({j: resfi})
    else:
        return resfi


parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched',
                  help="db name [%default]")
parser.add_option("--dbDir", type="str", default='', help="db dir [%default]")
parser.add_option("--outDir", type="str", default='',
                  help="output dir [%default]")
parser.add_option("--templateDir", type="str", default='',
                  help="template dir [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="healpix nside [%default]")
parser.add_option("--band", type="str", default='r', help="band [%default]")
parser.add_option("--nproc", type="int", default='8',
                  help="number of proc  [%default]")
parser.add_option("--fieldtype", type="str", default='DD',
                  help="field type DD or WFD[%default]")
parser.add_option("--x1", type="float", default='0.0',
                  help="SN x1 [%default]")
parser.add_option("--color", type="float", default='0.0',
                  help="SN color [%default]")
parser.add_option("--zmax", type="float", default='1.2',
                  help="zmax for simu [%default]")
parser.add_option("--dithering", type="int", default='0',
                  help="dithering for DDF [%default]")
# parser.add_option("--overlap", type="float", default='0.9',
#                  help="overlap focal plane/pixel [%default]")
parser.add_option("--simuType", type="int", default='0',
                  help="flag for new simulations [%default]")
parser.add_option("--saveData", type="int", default='0',
                  help="flag to dump data on disk [%default]")
parser.add_option("--metric", type="str", default='cadence',
                  help="metric to process [%default]")
parser.add_option("--coadd", type="int", default='1',
                  help="nightly coaddition [%default]")
parser.add_option("--nodither", type="str", default='',
                  help="to remove dithering - for DDF only[%default]")
parser.add_option("--ramin", type=float, default=0.,
                  help="ra min for obs area - for WDF only[%default]")
parser.add_option("--ramax", type=float, default=360.,
                  help="ra max for obs area - for WDF only[%default]")
parser.add_option("--decmin", type=float, default=-1.,
                  help="dec min for obs area - for WDF only[%default]")
parser.add_option("--decmax", type=float, default=-1.,
                  help="dec max for obs area - for WDF only[%default]")


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
simuType = opts.simuType
outDir = opts.outDir
templateDir = opts.templateDir
saveData = opts.saveData
metric = opts.metric
coadd = opts.coadd
nodither = opts.nodither
ramin = opts.ramin
ramax = opts.ramax
decmin = opts.decmin
decmax = opts.decmax

pixArea = hp.nside2pixarea(nside, degrees=True)
if outDir == '':
    outDir = 'MetricOutput'
if templateDir == '':
    templateDir = '/sps/lsst/data/dev/pgris/Templates_final_new'

outputDir = '{}/{}{}'.format(outDir,dbName,nodither)
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)
# List of (instance of) metrics to process
metricList = []
"""
if band != 'all':
    metricList.append(SNRMetricWrapper(z=0.3))
"""
#metricList.append(CadenceMetricWrapper(season=seasons,fieldtype=fieldtype))
"""
# metricList.append(SNRRateMetricWrapper(z=0.3))
metricList.append(NSNMetricWrapper(fieldtype=fieldtype,
                                   pixArea=pixArea, season=-1,
                                   nside=nside, templateDir=templateDir,
                                   verbose=False, ploteffi=False))
"""
if metric == 'NSN':
    metricList.append(NSNMetricWrapper(fieldtype=fieldtype,
                                       pixArea=pixArea,season=-1,
                                       nside=nside, templateDir=templateDir,
                                       verbose=False, ploteffi=False,outeffi=True))

if metric == 'Cadence':
    metricList.append(CadenceMetricWrapper(season=-1,coadd=coadd,fieldtype=fieldtype,nside=nside,ramin=ramin,ramax=ramax,decmin=decmin,decmax=decmax))
if metric == 'SL':
    metricList.append(SLMetricWrapper(nside=nside,coadd=coadd,fieldtype=fieldtype))

# loading observations


observations = np.load('{}/{}.npy'.format(dbDir, dbName))
observations = renameFields(observations)
"""
print(observations.dtype)
stacker = CoaddStacker()

observations = stacker._run(observations)
"""

# print(obs.dtype)


"""
if fieldtype =='WFD':
    idx = observations['proposalId'] == 3
    observations = observations[idx]
"""

# fieldIds = [290]
# fieldIds = [290,1427, 2412, 2786]
# this is a "simple" tessalation using healpix
dictArea = {}
radius = 5.
if fieldtype == 'DD':
    if simuType > 0:
        fieldIds = [0]
    else:
        fieldIds = [290, 744, 1427, 2412, 2786]
    # fieldIds = [0]
    if simuType == 2:
        pixels = selectDD(observations, nside, fieldIds)
        print('hello ', pixels)
        observations = np.copy(pixels)
    else:
        observations = getFields(observations, fieldIds,simuType)
    # print(np.unique(observations['fieldId']))
    r = []
   
    if simuType == 1:
        r.append(('ELAIS', 744, 10.0, -45.52, radius))
    else:
        r.append(('ELAIS', 744, 0.0, -45.52, radius))
    r.append(('SPT', 290, 349.39, -63.32, radius))
    r.append(('COSMOS', 2786, 150.36, 2.84, radius))
    r.append(('XMM-LSS', 2412, 34.39, -5.09, radius))
    r.append(('CDFS', 1427, 53.00, -27.44, radius))
    areas = np.rec.fromrecords(
        r, names=['name', 'fieldId', 'Ra', 'Dec', 'radius'])

else:
    #pixels = pixelate(observations, nside, RaCol='fieldRA', DecCol='fieldDec')
    observations = getFields(observations, fieldIds=[1],simuType=simuType)
    minDec = decmin
    maxDec = decmax
    if decmin == -1.0 and decmax == -1.0:
        #in that case min and max dec are given by obs strategy
        minDec = np.min(observations['fieldDec'])-3.
        maxDec = np.max(observations['fieldDec'])+3.
    areas = pavingSky(ramin,ramax, minDec,maxDec, radius)
    #areas = pavingSky(20., 40., -40., -30., radius)
    print(observations.dtype)
    RaCol = 'fieldRA'
    DecCol = 'fieldDec'
    if simuType == 1:
        RaCol = 'Ra'
        DecCol = 'Dec'


    """
    if nodither == 'no':
        dir_csv = '/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db/2018-06-WPC/dithers/wp_descDithers_csvs/'
        csv_file = '{}/descDithers_{}.csv'.format(dir_csv,dbName)
        #load csv file in numpyarray
        dither = genfromtxt(csv_file, delimiter=',',dtype=[('Decdith', '<f8'), ('Radith', '<f8'), ('Rotdith', '<f8'),('ObsId', '<i8')])
        print(dither)
        dither = dither[1:]
        print(observations.dtype)
        observations.sort(order='observationId')
    """


        #print(test)

   

print('observations', len(observations),len(areas))



timeref = time.time()

# healpixels = np.unique(pixels[['healpixID','pixRa','pixDec']])
healpixels = areas
npixels = int(len(healpixels))
delta = npixels
if nproc > 1:
    delta = int(delta/(nproc))

tabpix = range(0, npixels, delta)
if npixels not in tabpix:
    tabpix = np.append(tabpix, npixels)

tabpix = tabpix.tolist()


if nproc >=7:
    if tabpix[-1]-tabpix[-2] <= 10:
        tabpix.remove(tabpix[-2])

print(tabpix, len(tabpix))
result_queue = multiprocessing.Queue()



print('observations',len(observations))
for j in range(len(tabpix)-1):
#for j in range(1,2):
    ida = tabpix[j]
    idb = tabpix[j+1]
    # print('go', ida, idb)
    # p = multiprocessing.Process(name='Subprocess-'+str(j), target=loop, args=(
    #    healpixels[ida:idb],band, metricList, shape,observations,j, result_queue))
    print('Field', healpixels[ida:idb])
    p = multiprocessing.Process(name='Subprocess-'+str(j), target=loop_area, args=(
        healpixels[ida:idb], band, metricList, observations, nside,outputDir,dbName, saveData,nodither,RaCol,DecCol,j, result_queue))
    p.start()


"""
resultdict = {}
for i in range(len(tabpix)-1):
    resultdict.update(result_queue.get())

for p in multiprocessing.active_children():
    p.join()
"""
"""
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
"""
#print('Done', time.time()-timeref)
