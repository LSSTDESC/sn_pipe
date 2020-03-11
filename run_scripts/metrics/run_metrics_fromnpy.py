# import matplotlib.pyplot as plt
#import matplotlib
# matplotlib.use('agg')
import numpy as np
from optparse import OptionParser
import time
import multiprocessing
import os
from numpy import genfromtxt
import sys
from metricWrapper import CadenceMetricWrapper, SNRMetricWrapper
from metricWrapper import ObsRateMetricWrapper, NSNMetricWrapper
from metricWrapper import SLMetricWrapper
from sn_tools.sn_obs import DataToPixels, ProcessPixels,renameFields,patchObs
from sn_tools.sn_io import getObservations

def processPatch(pointings, metricList, observations, nside, outDir, dbName, saveData=False, nodither=False, RACol='', DecCol='', j=0, output_q=None):

    print('processing area', j, pointings)

    #print('before stacker',observations[['observationStartMJD','filter','fieldRA','fieldDec','visitExposureTime']][:50])

    #observations.sort(order=['observationStartMJD'])
    #print(test)
    time_ref = time.time()
    ipoint = 1
 
    datapixels = DataToPixels(nside, RACol, DecCol, j, outDir, dbName, saveData)
    procpix = ProcessPixels(metricList,j,outDir=outDir, dbName=dbName, saveData=saveData)

    #print('eee',pointings)

    for index, pointing in pointings.iterrows():
        ipoint += 1
        print('pointing',ipoint)

        # get the pixels
        pixels= datapixels(observations, pointing['RA'], pointing['Dec'],
                            pointing['radius_RA'], pointing['radius_Dec'], ipoint, nodither, display=False)

        # select pixels that are inside the original area

        #idx = np.abs(pixels['pixRA']-pointing['RA'])<=pointing['radius_RA']/2.
        #idx &= np.abs(pixels['pixDec']-pointing['Dec'])<=pointing['radius_Dec']/2.
        """
        idx = (pixels['pixRA']-pointing['RA'])>=-pointing['radius_RA']/np.cos(pointing['radius_Dec'])/2.
        idx &= (pixels['pixRA']-pointing['RA'])<pointing['radius_RA']/np.cos(pointing['radius_Dec'])/2.
        """
        idx = (pixels['pixRA']-pointing['RA'])>=-pointing['radius_RA']/2.
        idx &= (pixels['pixRA']-pointing['RA'])<pointing['radius_RA']/2.
        idx &= (pixels['pixDec']-pointing['Dec'])>=-pointing['radius_Dec']/2.
        idx &= (pixels['pixDec']-pointing['Dec'])<pointing['radius_Dec']/2.

        
        print('cut',pointing['RA'],pointing['radius_RA'],pointing['Dec'],pointing['radius_Dec'])
        
        #datapixels.plot(pixels)
        print('after selection',len(pixels[idx]))
        procpix(pixels[idx],datapixels.observations,ipoint)

    print('end of processing for', j, time.time()-time_ref)

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
parser.add_option("--RAmin", type=float, default=0.,
                  help="RA min for obs area - for WDF only[%default]")
parser.add_option("--RAmax", type=float, default=360.,
                  help="RA max for obs area - for WDF only[%default]")
parser.add_option("--Decmin", type=float, default=-1.,
                  help="Dec min for obs area - for WDF only[%default]")
parser.add_option("--Decmax", type=float, default=-1.,
                  help="Dec max for obs area - for WDF only[%default]")
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
parser.add_option("--timer", type=int, default=0,
                  help="timer mode for the metric[%default]")
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

available_metrics = ['NSN', 'Cadence', 'SL', 'ObsRate', 'SNRr','SNRz']
if opts.metric not in available_metrics:
    print('Sorry to inform you that', opts.metric, 'is not a metric available')
    print('list of possible metrics:')
    print(available_metrics)
    sys.exit(0)

season_int = list(opts.seasons.split(','))
if season_int[0] == '-':
    season_int = -1
else:
    season_int = list(map(int, season_int))

metricname = opts.metric
if 'SNR' in opts.metric:
    metricname='SNR'

classname = '{}MetricWrapper'.format(metricname)

print('seasons and metric',season_int,metricname)

metricList.append(globals()[classname](name=opts.metric, season=season_int,
                                       coadd=opts.coadd, fieldType=opts.fieldType,
                                       nside=opts.nside,
                                       RAmin=opts.RAmin, RAmax=opts.RAmax,
                                       Decmin=opts.Decmin, Decmax=opts.Decmax,
                                       metadata=opts, outDir=outputDir))

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

    """
    idx = field['fieldName'] == 'SPT'
    if len(field[idx])>0:
    """
    p = multiprocessing.Process(name='Subprocess-'+str(j), target=processPatch, args=(
        healpixels[ida:idb], metricList, observations, opts.nside,
        outputDir, opts.dbName, opts.saveData, opts.remove_dithering, RACol, DecCol, j, result_queue))
    p.start()

