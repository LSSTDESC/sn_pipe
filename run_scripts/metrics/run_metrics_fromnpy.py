#import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('agg')
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
from sn_tools.sn_obs import pavingSky,getFields
import glob
from numpy import genfromtxt
from sn_tools.sn_io import Read_Sqlite

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
    print('hhh',RaCol,DecCol,saveData)
    myprocess = ProcessArea(nside,RaCol,DecCol,j,outDir,dbName,saveData)
    for pointing in pointings:
        ipoint += 1
        #print('pointing',ipoint)
        
       
        """
        myprocess = ProcessArea(
            nside, pointing['Ra'], pointing['Dec'], pointing['radius'], pointing['radius'], 'fieldRA', 'fieldDec',j,outDir,dbName)
        """
        #resdict = myprocess.process(observations, metricList,ipoint)
        
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
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
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
parser.add_option("--fieldType", type="str", default='DD',
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
dbExtens = opts.dbExtens
nside = opts.nside
band = opts.band
seasons = -1
nproc = opts.nproc
fieldType = opts.fieldType
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

outputDir = '{}/{}{}/{}'.format(outDir,dbName,nodither,metric)
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)
# List of (instance of) metrics to process
metricList = []

if metric == 'NSN':
    metricList.append(NSNMetricWrapper(fieldType=fieldType,
                                       pixArea=pixArea,season=-1,
                                       nside=nside, templateDir=templateDir,
                                       verbose=0, ploteffi=0,coadd=coadd,outputType='zlims',proxy_level=0,ramin=ramin,ramax=ramax))

if metric == 'Cadence':
    metricList.append(CadenceMetricWrapper(season=-1,coadd=coadd,fieldType=fieldType,nside=nside,ramin=ramin,ramax=ramax,decmin=decmin,decmax=decmax))
if metric == 'SL':
    metricList.append(SLMetricWrapper(nside=nside,coadd=coadd,fieldType=fieldType))
if metric == 'SNRRate':
    metricList.append(SNRRateMetricWrapper(nside=nside,coadd=coadd))

if 'SNR' in metric:
    band = metric[-1]
    metricList.append(SNRMetricWrapper(z=0.2,coadd=coadd,nside=nside,band=band,ramin=ramin,ramax=ramax,decmin=decmin,decmax=decmax))
"""
if metric == 'SNRz'
    metricList.append(SNRMetricWrapper(z=0.2,coadd=coadd,nside=nside,band='z',ramin=ramin,ramax=ramax,decmin=decmin,decmax=decmax))
"""
# loading observations

dbFullName = '{}/{}.{}'.format(dbDir, dbName,dbExtens)
# if extension is npy -> load
if dbExtens == 'npy':
    observations = np.load(dbFullName)
else:
    #db as input-> need to transform as npy
    print('looking for',dbFullName)
    keymap = {'observationStartMJD': 'mjd',
              'filter': 'band',
              'visitExposureTime': 'exptime',
              'skyBrightness': 'sky',
              'fieldRA': 'Ra',
              'fieldDec': 'Dec',}

    reader = Read_Sqlite(dbFullName)
    #sql = reader.sql_selection(None)
    observations = reader.get_data(cols=None, sql='',
                           to_degrees=False,
                           new_col_names=keymap)



observations = renameFields(observations)


# fieldIds = [290]
# fieldIds = [290,1427, 2412, 2786]
# this is a "simple" tessalation using healpix
dictArea = {}
radius = 10.
RaCol = 'fieldRA'
DecCol = 'fieldDec'
if 'Ra' in observations.dtype.names:
     RaCol = 'Ra'
     DecCol = 'Dec'

if fieldType == 'DD':
    fieldIds = [290,744,1427, 2412, 2786]
    observations = getFields(observations, fieldType, fieldIds,nside)
    r = []
   
    if simuType == 1:
        r.append(('ELAIS', 744, 10.0, -45.52, radius))
    else:
        r.append(('ELAIS', 744, 0.0, -45.52, radius))
    
    r.append(('COSMOS', 2786, 150.36, 2.84, radius))
    r.append(('XMM-LSS', 2412, 34.39, -5.09, radius))
    r.append(('CDFS', 1427, 53.00, -27.44, radius))
    if 'euclid' not in dbName:
        r.append(('SPT', 290, 349.39, -63.32, radius))
    else:
        r.append(('ADFS', 290, 61.00, -48.0, radius))

    areas = np.rec.fromrecords(
        r, names=['name', 'fieldId', 'Ra', 'Dec', 'radius'])    
 

else:
    if fieldType == 'WFD':
        observations = getFields(observations,'WFD')
        minDec = decmin
        maxDec = decmax
        if decmin == -1.0 and decmax == -1.0:
            #in that case min and max dec are given by obs strategy
            minDec = np.min(observations['fieldDec'])-3.
            maxDec = np.max(observations['fieldDec'])+3.
        areas = pavingSky(ramin,ramax, minDec,maxDec, radius)
        #areas = pavingSky(20., 40., -40., -30., radius)
        print(observations.dtype)
        
    if fieldType == 'Fake':
        #in that case: only one (Ra,Dec)
        radius = 0.1
        Ra = np.unique(observations[RaCol])[0]
        Dec = np.unique(observations[DecCol])[0]
        areas = pavingSky(Ra-radius/2.,Ra+radius/2.,Dec-radius/2.,Dec+radius/2.,radius)


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

print('observations', len(observations),len(areas))

timeref = time.time()

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

tabpix = np.linspace(1,npixels,nproc+1,dtype='int') 
print(tabpix, len(tabpix))
result_queue = multiprocessing.Queue()

print('observations',len(observations))
for j in range(len(tabpix)-1):
#for j in range(7,8):
    ida = tabpix[j]
    idb = tabpix[j+1]

    print('Field', healpixels[ida:idb])
    p = multiprocessing.Process(name='Subprocess-'+str(j), target=loop_area, args=(
        healpixels[ida:idb], band, metricList, observations, nside,outputDir,dbName, saveData,nodither,RaCol,DecCol,j, result_queue))
    p.start()
