# import matplotlib.pyplot as plt
#import matplotlib
# matplotlib.use('agg')
import numpy as np
from optparse import OptionParser
import time
import multiprocessing
import os
import glob
import sys
import random
import pandas as pd
from metricWrapper import CadenceMetricWrapper, SNRMetricWrapper
from metricWrapper import ObsRateMetricWrapper, NSNMetricWrapper
from metricWrapper import SLMetricWrapper
from sn_tools.sn_obs import DataToPixels, ProcessPixels, renameFields, patchObs
from sn_tools.sn_io import getObservations


class processMetrics:
    def __init__(self, dbDir, dbName, dbExtens,
                 fieldType, nside,
                 RAmin, RAmax,
                 Decmin, Decmax,
                 saveData, remove_dithering,
                 outDir, nprocs, metricList,
                 pixelmap_dir='', npixels=0, nclusters=5, radius=4.):
        """
        Class to process data ie run metrics on a set of pixels

        Parameters
        --------------
        dbDir: str
          dir location of observing strategy file
        dbName: str
           observing strategy name
        dbExtens: str
           database extension (npy, db, ...)
        fieldType: str
            type of field: DD, WFD, Fake
        nside: int
           healpix nside parameter
        RAmin: float
          min RA of the area to process
        RAmax: float
          max RA of the area to process
       Decmin: float
          min Dec of the area to process
        Decmax: float
          max Dec of the area to process
        saveData: bool
          to save ouput data or not
        remove_dithering: bool
          to remove dithering (to use for DD studies)
        outDir: str
          output directory location
        nprocs: int,
          number of cores to run
        metricList: list(metrics)
           list of metrics to process
        pixelmap_dir: str, opt
           location directory where maps pixels<->observations are (default: '')
        npixels: int, opt
           number of pixels to run on (default: 0)

        """

        self.dbDir = dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
        self.fieldType = fieldType
        self.nside = nside
        self.RAmin = RAmin
        self.RAmax = RAmax
        self.Decmin = Decmin
        self.Decmax = Decmax
        self.saveData = saveData
        self.remove_dithering = remove_dithering
        self.outDir = outDir
        self.nprocs = nprocs
        self.metricList = metricList
        self.pixelmap_dir = pixelmap_dir
        self.npixels = npixels
        self.nclusters = nclusters
        self.radius = radius
        observations, patches = self.load()

        # select observation in this area
        idx = observations[self.RACol] >= RAmin-5.
        idx &= observations[self.RACol] < RAmax+5.
        idx &= observations[self.DecCol] >= Decmin-5.
        idx &= observations[self.DecCol] < Decmax+5.

        print('before', len(observations), RAmin, RAmax, Decmin, Decmax)
        #observations = observations[idx]
        print('after', len(observations))

        """
        import matplotlib.pyplot as plt
        plt.plot(observations[self.RACol],observations[self.DecCol],'ko')
        plt.show()
        """

        if self.pixelmap_dir == '':
            self.multiprocess(patches, observations, self.processPatch)
        else:
            # load the pixel maps
            print('pixel map loading', self.pixelmap_dir, self.fieldType,
                  self.nside, self.dbName, self.npixels)
            search_path = '{}/{}/{}_{}_nside_{}_{}_{}_{}_{}.npy'.format(self.pixelmap_dir,
                                                                        self.dbName, self.dbName,
                                                                        self.fieldType, self.nside,
                                                                        self.RAmin, self.RAmax,
                                                                        self.Decmin, self.Decmax)
            print(search_path)
            pixelmap_files = glob.glob(search_path)
            print(pixelmap_files)
            if not pixelmap_files:
                print('Severe problem: pixel map does not exist!!!!')
            else:
                self.pixelmap = np.load(pixelmap_files[0])
                if self.npixels == -1:
                    self.npixels = len(np.unique(self.pixelmap['healpixID']))
                random_pixels = self.randomPixels()
                print('number of pixels to process', len(random_pixels))
                self.multiprocess(random_pixels, observations,
                                  self.processPixels)

    def load(self):
        """
        Method to load observations and patches dims on the sky

        Returns
        ------------
       observations: numpy array
         numpy array with observations
       patches: pandas df
        patches coordinates on the sky
        """

        # loading observations

        observations = getObservations(self.dbDir, self.dbName, self.dbExtens)

        # rename fields

        observations = renameFields(observations)

        self.RACol = 'fieldRA'
        self.DecCol = 'fieldDec'

        if 'RA' in observations.dtype.names:
            self.RACol = 'RA'
            self.DecCol = 'Dec'

        observations, patches = patchObs(observations, self.fieldType,
                                         self.dbName,
                                         self.nside,
                                         self.RAmin, self.RAmax,
                                         self.Decmin, self.Decmax,
                                         self.RACol, self.DecCol,
                                         display=False,
                                         nclusters=self.nclusters, radius=self.radius)

        return observations, patches

    def multiprocess(self, patches, observations, func):
        """
        Method to perform multiprocessing of metrics

        Parameters
        ---------------
        patches: pandas df
          patches coordinates on the sky
        observations: numpy array
         numpy array with observations

        """

        timeref = time.time()

        healpixels = patches
        npixels = int(len(healpixels))

        tabpix = np.linspace(0, npixels, self.nprocs+1, dtype='int')
        print(tabpix, len(tabpix))
        result_queue = multiprocessing.Queue()

        print('in multi process', npixels, self.nprocs)
        # multiprocessing
        for j in range(len(tabpix)-1):
            ida = tabpix[j]
            idb = tabpix[j+1]

            print('Field', j, len(healpixels[ida:idb]))

            field = healpixels[ida:idb]

            """
            idx = field['fieldName'] == 'SPT'
            if len(field[idx]) > 0:
            """
            p = multiprocessing.Process(name='Subprocess-'+str(j), target=func, args=(
                healpixels[ida:idb], observations, j, result_queue))
            p.start()

    def processPatch(self, pointings, observations, j=0, output_q=None):
        """
        Method to process a patch

        Parameters
        --------------
        pointings: numpy array
          array with a set of area on the sky
        observations: numpy array
           array of observations
        j: int, opt
          index number of multiprocessing (default: 0)
        output_q: multiprocessing.Queue(), opt
          queue of the multiprocessing (default: None)
        """

        print('processing area', j, pointings)

        time_ref = time.time()
        ipoint = 1

        datapixels = DataToPixels(
            self.nside, self.RACol, self.DecCol, j, self.outDir, self.dbName)

        procpix = ProcessPixels(
            self.metricList, j, outDir=self.outDir, dbName=self.dbName, saveData=self.saveData)

        for index, pointing in pointings.iterrows():
            ipoint += 1
            print('pointing', ipoint)

            # print('there man', np.unique(observations[[self.RACol, self.DecCol]]), pointing[[
            #      'RA', 'Dec', 'radius_RA', 'radius_Dec']])
            # get the pixels
            pixels = datapixels(observations, pointing['RA'], pointing['Dec'],
                                pointing['radius_RA'], pointing['radius_Dec'], ipoint, self.remove_dithering, display=False)

            # select pixels that are inside the original area

            pixels_run = pixels
            if self.fieldType != 'Fake' and self.fieldType != 'DD':
                idx = (pixels['pixRA']-pointing['RA']) >= - \
                    pointing['radius_RA']/2.
                idx &= (pixels['pixRA']-pointing['RA']
                        ) < pointing['radius_RA']/2.
                idx &= (pixels['pixDec']-pointing['Dec']) >= - \
                    pointing['radius_Dec']/2.
                idx &= (pixels['pixDec']-pointing['Dec']
                        ) < pointing['radius_Dec']/2.
                pixels_run = pixels[idx]

            print('cut', pointing['RA'], pointing['radius_RA'],
                  pointing['Dec'], pointing['radius_Dec'])

            # datapixels.plot(pixels)
            print('after selection', len(pixels_run), datapixels.observations)
            procpix(pixels_run, datapixels.observations, ipoint)

        print('end of processing for', j, time.time()-time_ref)

    def processPixels(self, pixels, observations, j=0, output_q=None):
        """
        Method to process a pixel

        Parameters
        --------------
        pointings: numpy array
          array with a set of area on the sky
        observations: numpy array
           array of observations
        j: int, opt
          index number of multiprocessing (default: 0)
        output_q: multiprocessing.Queue(), opt
          queue of the multiprocessing (default: None)
        """
        time_ref = time.time()
        procpix = ProcessPixels(
            self.metricList, j, outDir=self.outDir, dbName=self.dbName, saveData=self.saveData)

        valsdf = pd.DataFrame(self.pixelmap)
        ido = valsdf['healpixID'].isin(pixels)
        procpix(valsdf[ido], observations, self.npixels)

        print('end of processing for', j, time.time()-time_ref)

    def randomPixels(self):
        """
        Method to choose a random set of pixels


        Returns
        ---------
       healpixIDs: list of randomly chosen healpix IDs

        """

        hIDs = np.unique(self.pixelmap['healpixID'])
        healpixIDs = random.sample(hIDs.tolist(), self.npixels)

        return healpixIDs

    def randomPixels_old(self, pixeldict, npix_tot):
        """
        Method to choose a random set of pixels

        Parameters
        ---------------
        pixeldict: dict
          dict with int as keys and tabas items
          where tab is an array of healpix ID

        Returns
        ---------
        random_pixels: pandas df
          selected pixels from random choice

        """
        # get random pixels

        random_pixels = pd.DataFrame()
        for (key, vals) in pixeldict.items():
            hIDs = np.unique(vals['healpixID'])
            frac = len(hIDs)/npix_tot
            nrandom = int(self.npixels*frac)
            healpixIDs = random.sample(hIDs.tolist(), nrandom)
            print(healpixIDs, type(healpixIDs))
            valsdf = pd.DataFrame(vals)
            ido = valsdf['healpixID'].isin(healpixIDs)
            random_pixels = pd.concat((random_pixels, valsdf[ido]), sort=False)

        return healpixIDs


def processPatch(pointings, metricList, observations, nside, outDir, dbName, saveData=False, nodither=False, RACol='', DecCol='', j=0, output_q=None):

    print('processing area', j, pointings)

    #print('before stacker',observations[['observationStartMJD','filter','fieldRA','fieldDec','visitExposureTime']][:50])

    # observations.sort(order=['observationStartMJD'])
    # print(test)
    time_ref = time.time()
    ipoint = 1

    datapixels = DataToPixels(nside, RACol, DecCol, j,
                              outDir, dbName, saveData)
    procpix = ProcessPixels(metricList, j, outDir=outDir,
                            dbName=dbName, saveData=saveData)

    # print('eee',pointings)

    for index, pointing in pointings.iterrows():
        ipoint += 1
        print('pointing', ipoint)

        # get the pixels
        pixels = datapixels(observations, pointing['RA'], pointing['Dec'],
                            pointing['radius_RA'], pointing['radius_Dec'], ipoint, nodither, display=False)

        # select pixels that are inside the original area

        #idx = np.abs(pixels['pixRA']-pointing['RA'])<=pointing['radius_RA']/2.
        #idx &= np.abs(pixels['pixDec']-pointing['Dec'])<=pointing['radius_Dec']/2.
        """
        idx = (pixels['pixRA']-pointing['RA'])>=-pointing['radius_RA']/np.cos(pointing['radius_Dec'])/2.
        idx &= (pixels['pixRA']-pointing['RA'])<pointing['radius_RA']/np.cos(pointing['radius_Dec'])/2.
        """
        idx = (pixels['pixRA']-pointing['RA']) >= -pointing['radius_RA']/2.
        idx &= (pixels['pixRA']-pointing['RA']) < pointing['radius_RA']/2.
        idx &= (pixels['pixDec']-pointing['Dec']) >= -pointing['radius_Dec']/2.
        idx &= (pixels['pixDec']-pointing['Dec']) < pointing['radius_Dec']/2.

        print('cut', pointing['RA'], pointing['radius_RA'],
              pointing['Dec'], pointing['radius_Dec'])

        # datapixels.plot(pixels)
        print('after selection', len(pixels[idx]))
        procpix(pixels[idx], datapixels.observations, ipoint)

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
parser.add_option("--dirFake", type=str, default='input/Fake_cadence',
                  help="dir of fake files for the metric[%default]")
parser.add_option("--names_ref", type=str, default='SNCosmo',
                  help="ref name for the ref files for the metric[%default]")
parser.add_option("--x1", type=float, default=-2.0,
                  help="Supernova stretch[%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="Supernova color[%default]")
parser.add_option("--pixelmap_dir", type=str, default='',
                  help="dir where to find pixel maps[%default]")
parser.add_option("--npixels", type=int, default=0,
                  help="number of pixels to process[%default]")
parser.add_option("--nclusters", type=int, default=0,
                  help="number of clusters in data (DD only)[%default]")
parser.add_option("--radius", type=float, default=4.,
                  help="radius around clusters (DD and Fakes)[%default]")


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

available_metrics = ['NSN', 'Cadence', 'SL', 'ObsRate', 'SNRr', 'SNRz']
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
    metricname = 'SNR'

classname = '{}MetricWrapper'.format(metricname)


metricList.append(globals()[classname](name=opts.metric, season=season_int,
                                       coadd=opts.coadd, fieldType=opts.fieldType,
                                       nside=opts.nside,
                                       RAmin=opts.RAmin, RAmax=opts.RAmax,
                                       Decmin=opts.Decmin, Decmax=opts.Decmax,
                                       npixels=opts.npixels, metadata=opts, outDir=outputDir))

print('seasons and metric', season_int,
      metricname, opts.pixelmap_dir, opts.npixels)
process = processMetrics(opts.dbDir, opts.dbName, opts.dbExtens,
                         opts.fieldType, opts.nside,
                         opts.RAmin, opts.RAmax,
                         opts.Decmin, opts.Decmax,
                         opts.saveData, opts.remove_dithering,
                         outputDir, opts.nproc, metricList,
                         opts.pixelmap_dir, opts.npixels,
                         opts.nclusters, opts.radius)


"""
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
        healpixels[ida:idb], metricList, observations, opts.nside,
        outputDir, opts.dbName, opts.saveData, opts.remove_dithering, RACol, DecCol, j, result_queue))
    p.start()

"""
