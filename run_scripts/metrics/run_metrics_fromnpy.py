# import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('agg')
import numpy as np
from metricWrapper import CadenceMetricWrapper, SNRMetricWrapper
from metricWrapper import ObsRateMetricWrapper, NSNMetricWrapper
from metricWrapper import SLMetricWrapper
from sn_tools.sn_obs import renameFields, pixelate, GetShape, ObsPixel, ProcessArea
from sn_tools.sn_obs import season as seasoncalc
from optparse import OptionParser
import time
import multiprocessing
import os
import numpy.lib.recfunctions as rf
from sn_stackers.coadd_stacker import CoaddStacker
from sn_tools.sn_obs import pavingSky, getFields
import glob
from numpy import genfromtxt
from sn_tools.sn_io import getObservations
from sn_tools.sn_cadence_tools import ClusterObs
import yaml
import sys


def loop_area(pointings, metricList, observations, nside, outDir, dbName, saveData, nodither, RaCol, DecCol, j=0, output_q=None):

    resfi = {}
    # print(np.unique(observations[['fieldRA', 'fieldDec']]))

    print('processing pointings', j, pointings)

    for metric in metricList:
        resfi[metric.name] = None
        listf = glob.glob('{}/*_{}_{}*'.format(outDir, metric.name, j))
        if len(listf) > 0:
            for val in listf:
                os.system('rm {}'.format(val))

    # print(test)
    time_ref = time.time()
    # print('Starting processing', len(pointings),j)
    ipoint = 1
    # myprocess = ProcessArea(nside,'fieldRA', 'fieldDec',j,outDir,dbName,saveData)
    print('hhh', RaCol, DecCol, saveData)
    myprocess = ProcessArea(nside, RaCol, DecCol, j, outDir, dbName, saveData)
    for pointing in pointings:
        ipoint += 1
        # print('pointing',ipoint)

        """
        myprocess = ProcessArea(
            nside, pointing['Ra'], pointing['Dec'], pointing['radius'], pointing['radius'], 'fieldRA', 'fieldDec',j,outDir,dbName)
        """
        # resdict = myprocess.process(observations, metricList,ipoint)

        resdict = myprocess(observations, metricList, pointing['Ra'], pointing['Dec'],
                            pointing['radius'], pointing['radius'], ipoint, nodither, display=False)

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
    print('end of processing for', j, time.time()-time_ref)

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
parser.add_option("--dbDir", type="str",
                  default='/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db', help="db dir [%default]")
parser.add_option("--outDir", type="str", default='MetricOutput',
                  help="output dir [%default]")
parser.add_option("--templateDir", type="str", default='/sps/lsst/data/dev/pgris/Templates_final_new',
                  help="template dir [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="healpix nside [%default]")
parser.add_option("--nproc", type="int", default='8',
                  help="number of proc  [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type DD or WFD[%default]")
parser.add_option("--zmax", type="float", default='1.2',
                  help="zmax for simu [%default]")
parser.add_option("--remove_dithering", type="int", default='0',
                  help="remove dithering for DDF [%default]")
parser.add_option("--simuType", type="int", default='0',
                  help="flag for new simulations [%default]")
parser.add_option("--saveData", type="int", default='0',
                  help="flag to dump data on disk [%default]")
parser.add_option("--metric", type="str", default='cadence',
                  help="metric to process [%default]")
parser.add_option("--coadd", type="int", default='1',
                  help="nightly coaddition [%default]")
# parser.add_option("--nodither", type="str", default='',
#                  help="to remove dithering - for DDF only[%default]")
parser.add_option("--ramin", type=float, default=0.,
                  help="ra min for obs area - for WDF only[%default]")
parser.add_option("--ramax", type=float, default=360.,
                  help="ra max for obs area - for WDF only[%default]")
parser.add_option("--decmin", type=float, default=-1.,
                  help="dec min for obs area - for WDF only[%default]")
parser.add_option("--decmax", type=float, default=-1.,
                  help="dec max for obs area - for WDF only[%default]")
parser.add_option("--proxy_level", type=int, default=0,
                  help="proxy level for the metric[%default]")
parser.add_option("--T0s", type=str, default='all',
                  help="T0 values to consider[%default]")
parser.add_option("--lightOutput", type=int, default=0,
                  help="light LC output[%default]")
parser.add_option("--outputType", type=str, default='zlims',
                  help="outputType of the metric[%default]")
parser.add_option("--seasons", type=str, default='-1',
                  help="seasons to process[%default]")
parser.add_option("--verbose", type=int, default=0,
                  help="verbose mode for the metric[%default]")
parser.add_option("--ploteffi", type=int, default=0,
                  help="plot efficiencies for the metric[%default]")
parser.add_option("--z", type=float, default=0.3,
                  help="redshift for the metric[%default]")
parser.add_option("--band", type=str, default='r',
                  help="band for the metric[%default]")
parser.add_option("--dirRefs", type=str, default='reference_files',
                  help="dir of reference files for the metric[%default]")
parser.add_option("--dirFakes", type=str, default='input/Fake_cadence',
                  help="dir of fake files for the metric[%default]")
parser.add_option("--names_ref", type=str, default='SNCosmo',
                  help="ref name for the ref files for the metric[%default]")
parser.add_option("--x1", type=float, default=-2.0,
                  help="Supernova stretch[%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="Supernova color[%default]")

opts, args = parser.parse_args()

print('Start processing...', opts)


# prepare outputDir
nodither = ''
if opts.remove_dithering:
    nodither = '_nodither'
outputDir = '{}/{}{}/{}'.format(opts.outDir,
                                opts.dbName, nodither, opts.metric)
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

# List of (instance of) metrics to process
metricList = []

# check whether the metric is available

available_metrics = ['NSN', 'Cadence', 'SL', 'ObsRate', 'SNR']
if opts.metric not in available_metrics:
    print('Sorry to inform you that', metric, 'is not a metric available')
    print('list of possible metrics:')
    print(available_metrics)
    sys.exit(0)

season_int = list(opts.seasons.strip())
if season_int[0] == '-':
    season_int = -1
else:
    season_int = list(map(int, season_int))

classname = '{}MetricWrapper'.format(opts.metric)

metricList.append(globals()[classname](name=opts.metric, season=season_int,
                                       coadd=opts.coadd, fieldType=opts.fieldType,
                                       nside=opts.nside,
                                       ramin=opts.ramin, ramax=opts.ramax,
                                       decmin=opts.decmin, decmax=opts.decmax,
                                       metadata=opts, outDir=outputDir))

# loading observations

observations = getObservations(opts.dbDir, opts.dbName, opts.dbExtens)

# rename fields

observations = renameFields(observations)

dictArea = {}
radius = 10.
RaCol = 'fieldRA'
DecCol = 'fieldDec'
if 'Ra' in observations.dtype.names:
    RaCol = 'Ra'
    DecCol = 'Dec'

if opts.fieldType == 'DD':
    nclusters = 5
    if 'euclid' in opts.dbName:
        nclusters = 6

    print(np.unique(observations['fieldId']))
    fieldIds = [290, 744, 1427, 2412, 2786]
    observations = getFields(
        observations, opts.fieldType, fieldIds, opts.nside)

    print('before cluster', len(observations))
    # get clusters out of these obs
    radius = 3.0

    clusters = ClusterObs(
        observations, nclusters=nclusters, dbName=opts.dbName).clusters

    clusters = rf.append_fields(clusters, 'radius', [radius]*len(clusters))

    areas = rf.rename_fields(clusters, {'RA': 'Ra'})


else:
    if opts.fieldType == 'WFD':
        observations = getFields(observations, 'WFD')
        minDec = opts.decmin
        maxDec = opts.decmax
        if minDec == -1.0 and maxDec == -1.0:
            # in that case min and max dec are given by obs strategy
            minDec = np.min(observations['fieldDec'])-3.
            maxDec = np.max(observations['fieldDec'])+3.
        areas = pavingSky(opts.ramin, opts.ramax, minDec, maxDec, radius)
        # areas = pavingSky(20., 40., -40., -30., radius)
        print(observations.dtype)

    if opts.fieldType == 'Fake':
        # in that case: only one (Ra,Dec)
        radius = 0.1
        Ra = np.unique(observations[RaCol])[0]
        Dec = np.unique(observations[DecCol])[0]
        areas = pavingSky(Ra-radius/2., Ra+radius/2., Dec -
                          radius/2., Dec+radius/2., radius)


print('observations', len(observations), len(areas))

timeref = time.time()

healpixels = areas
npixels = int(len(healpixels))


tabpix = np.linspace(0, npixels, opts.nproc+1, dtype='int')
print(tabpix, len(tabpix))
result_queue = multiprocessing.Queue()

print('observations', len(observations))
for j in range(len(tabpix)-1):
    # for j in range(7,8):
    ida = tabpix[j]
    idb = tabpix[j+1]

    print('Field', healpixels[ida:idb])
    p = multiprocessing.Process(name='Subprocess-'+str(j), target=loop_area, args=(
        healpixels[ida:idb], metricList, observations, opts.nside,
        outputDir, opts.dbName, opts.saveData, opts.remove_dithering, RaCol, DecCol, j, result_queue))
    p.start()
