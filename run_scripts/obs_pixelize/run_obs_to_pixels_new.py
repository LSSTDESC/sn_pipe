import numpy as np
from optparse import OptionParser
import time
import multiprocessing
import os
from sn_tools.sn_obs import DataToPixels_new, ProcessPixels, renameFields, patchObs
from sn_tools.sn_obs import renameDDF
from sn_tools.sn_io import getObservations
import pandas as pd
from os.path import exists


def skyPatch(RAmin, RAmax, nRA, Decmin, Decmax, nDec, outName):
    """
    Method to define patches on the sky

    Parameters
    --------------
    RAmin: float
      min RA of the area to be paved
    RAmax: float
      max RA of the area to be paved
    nRA: int
      number of paves in RA
   Decmin: float
      min Dec of the area to be paved
    Decmax: float
      max Dec of the area to be paved
    nDec: int
      number of paves in Dec
    outName: str
      name of the outputfile where the resulting numpy array is stored

    """

    RA = np.linspace(RAmin, RAmax, nRA+1)
    Dec = np.linspace(Decmin, Decmax, nDec+1)

    r = []
    for iRA in range(len(RA)-1):
        for iDec in range(len(Dec)-1):
            r.append((RA[iRA], RA[iRA+1], Dec[iDec], Dec[iDec+1]))

    skyPatch = np.rec.fromrecords(
        r, names=['RAmin', 'RAmax', 'Decmin', 'Decmax'])

    print(skyPatch)

    np.save(outName, skyPatch)


class procObsPixels:
    def __init__(self, outDir, dbDir, dbName, dbExtens, nodither,
                 fieldType, RAmin, RAmax, Decmin, Decmax, nside,
                 nprocs, saveData=False, fieldName='unknown',
                 radius=4, project_FP='gnomonic', VRO_FP='circle', telrot=0):
        """
        Class to process obs <-> pixels on a patch of the sky

        Parameters
        --------------
         outDir: str
           output dir name
         dbDir: str
           dir where the observing strategy db are located
         dbName: str
            name of the observing strategy
         dbExtens: str
           extension (db or npy) of the observing strategy file
         nodither: bool
            to remove dithering
         fieldType: str
           type of field (DD, WFD, Fakes) considered
         RAmin: float
          min RA of the patch
         RAmax: float
          max RA of the patch
         Decmin: float
          min Dec of the patch
         Decmax: float
           max Dec of the patch
         nside: int
           nside parameter for healpix
        nprocs: int
           number of procs when processing
        saveData: bool, opt
           to save the data (default:False)
        radius: float, opt
         radius around DD center to grab pixels
        project_FP: str, opt
          type of projection on FP (gnomonic or hp_query) (default: gnomonic)
        VRO_FP: str, opt
          geometry of the VRO FP (circle or realistic) (default: circle)
        telrot: int, opt
          factor to apply to the telescope rotation angle

        """

        # load class parameters
        self.dbDir = dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
        self.nodither = nodither
        self.fieldType = fieldType
        self.fieldName = fieldName
        self.RAmin = RAmin
        self.RAmax = RAmax
        self.Decmin = Decmin
        self.Decmax = Decmax
        self.nside = nside
        self.nprocs = nprocs
        self.saveData = saveData
        self.radius = radius
        self.project_FP = project_FP
        self.VRO_FP = VRO_FP
        self.telrot = telrot

        # create output directory
        self.genDir(outDir)

        self.outName = {}

        if self.fieldType == 'DD':
            fieldName = fieldName.split(',')
            for field in fieldName:
                self.outName[field] = '{}/{}_{}_nside_{}_{}_{}_{}_{}_{}.npy'.format(self.outDir,
                                                                                    dbName, fieldType, nside,
                                                                                    RAmin, RAmax,
                                                                                    Decmin, Decmax,
                                                                                    field)
        if self.fieldType == 'WFD':
            self.outName['WFD'] = '{}/{}_{}_nside_{}_{}_{}_{}_{}.npy'.format(self.outDir,
                                                                             dbName, fieldType, nside,
                                                                             RAmin, RAmax,
                                                                             Decmin, Decmax)

        # if the files already exist -> do not re-process it

    def __call__(self):

        # observations, patches = self.load_obs()

        observations = self.get_obs()

        datapixels = DataToPixels_new(
            self.nside, self.project_FP, self.VRO_FP, telrot=self.telrot, nproc=self.nprocs)

        print('proccesssi', np.unique(
            observations['note']), len(observations))

        # run using multiprocessing
        # self.multiprocess(patches, observations)
        if self.fieldType == 'WFD':
            pixels = datapixels(observations, display=False)
            pixels['fieldName'] = 'WFD'
            if self.saveData:
                np.save(self.outName['WFD'], pixels.to_records(index=False))
        if self.fieldType == 'DD':
            for i, field in enumerate(self.fieldName.split(',')):
                idx = observations['note'] == field
                pixels = datapixels(np.copy(observations[idx]), display=False)
                pixels['fieldName'] = field
                if self.saveData:
                    np.save(self.outName[field], res.to_records(index=False))

    def genDir(self, outDir):
        """
        Method to create directory
        """

        # prepare outputDir
        nodither = ''
        if self.nodither:
            nodither = '_nodither'

        self.outDir = '{}/{}{}'.format(outDir,
                                       self.dbName, nodither)

        # create output directory (if necessary)

        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)

    def get_obs(self):

        # loading all obs here
        observations = load_obs(self.dbDir, self.dbName, self.dbExtens)
        self.RACol = 'fieldRA'
        self.DecCol = 'fieldDec'

        if 'RA' in observations.dtype.names:
            self.RACol = 'RA'
            self.DecCol = 'Dec'

        if 'note' in observations.dtype.names:
            ido = np.core.defchararray.find(
                observations['note'].astype(str), 'DD')
            if ido.tolist():
                ies = np.ma.asarray(
                    list(map(lambda st: False if st != -1 else True, ido)))
                if self.fieldType == 'WFD':
                    return observations[ies]
                if self.fieldType == 'DD':
                    return renameDDF(observations[~ies], self.RACol, self.DecCol)
        else:
            return observations

    def processPatch(self, pointings, observations, j=0, output_q=None):
        """
        Method to process a single patch on the sky

        Parameters
        --------------
        pointings:
        observations: numpy array
            array of observations
        j: int, opt
           internal parameter (proc ID) (default: 0)
        output_q: multiprocessing.Queue(), opt
           queue for multi processing (default: None)

        Returns
        ----------
        Two possibilities:
          - output_q is None: pandas df of the results
          - output_queue is not None: pandas df of the results inside a dict with key j


        """

        print('processing area', j, len(pointings))

        time_ref = time.time()
        ipoint = 1

        datapixels = DataToPixels_new(
            self.nside, self.project_FP, self.VRO_FP, self.telrot, nproc=self.nprocs)

        pixelsTot = pd.DataFrame()
        print('starting process', j)
        for index, pointing in pointings.iterrows():
            ipoint += 1
            print('pointing', ipoint)

            # get the pixels
            pixels = datapixels(observations, display=False)

            """
            import matplotlib.pyplot as plt
            print(pixels.columns, len(
                np.unique(observations['observationId'])))
            plt.plot(
                observations[self.RACol], observations[self.DecCol], color='k', marker='o', mfc=None, linestyle='None')
            plt.plot(pixels['pixRA'], pixels['pixDec'], 'r*')
            plt.show()
            """
            # print(test)
            sel = pixels
            if pixels is not None:
                if self.fieldType == 'DD':
                    pixels['fieldName'] = pointing['fieldName']
                if self.fieldType == 'WFD':
                    # select pixels that are inside the original area
                    pixels['fieldName'] = 'WFD'

                pixelsTot = pd.concat((pixelsTot, sel), sort=False)
        print('end of processing for', j, time.time()-time_ref)

        if output_q is not None:
            return output_q.put({j: pixelsTot})
        else:
            return pixelsTot


def load_obs(dbDir, dbName, dbExtens):
    """
    Method to load observations

    Returns
    ----------
    observations: numpy array of observations

    """

    # loading observations

    observations = getObservations(
        dbDir, dbName, dbExtens)

    # rename fields

    observations = renameFields(observations)

    return observations


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
parser.add_option("--nprocs", type="int", default='8',
                  help="number of procs  [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type DD or WFD[%default]")
parser.add_option("--remove_dithering", type="int", default='0',
                  help="remove dithering for DDF [%default]")
parser.add_option("--simuType", type="int", default='0',
                  help="flag for new simulations [%default]")
parser.add_option("--saveData", type="int", default='0',
                  help="flag to dump data on disk [%default]")
parser.add_option("--RAmin", type=float, default=0.,
                  help="RA min for obs area - [%default]")
parser.add_option("--RAmax", type=float, default=360.,
                  help="RA max for obs area - [%default]")
parser.add_option("--nRA", type=int, default=1,
                  help="number of RA patches - [%default]")
parser.add_option("--Decmin", type=float, default=-1.,
                  help="Dec min for obs area - [%default]")
parser.add_option("--Decmax", type=float, default=-1.,
                  help="Dec max for obs area - [%default]")
parser.add_option("--nDec", type=int, default=1,
                  help="number of Dec patches - [%default]")
parser.add_option("--verbose", type=int, default=0,
                  help="verbose mode for the metric[%default]")
parser.add_option("--fieldName", type='str', default='WFD',
                  help="fieldName [%default]")
parser.add_option("--nclusters", type=int, default=6,
                  help="number of clusters - for DD only[%default]")
parser.add_option("--radius", type=float, default=4.,
                  help="radius around center - for DD only[%default]")
parser.add_option("--VRO_FP", type=str, default='circular',
                  help="VRO Focal Plane (circle or realistic) [%default]")
parser.add_option("--project_FP", type=str, default='gnomonic',
                  help="Focal Plane projection (gnomonic or hp_query) [%default]")
parser.add_option("--telrot", type=int, default=0,
                  help="telescope rotation angle [%default]")

opts, args = parser.parse_args()

print('Start processing...', opts)

if not os.path.isdir(opts.outDir):
    os.makedirs(opts.outDir)

skymapName = '{}/skypatch_{}_{}_{}_{}_{}_{}.npy'.format(opts.outDir,
                                                        opts.RAmin, opts.RAmax, opts.nRA, opts.Decmin, opts.Decmax, opts.nDec)

if opts.fieldType == 'WFD':
    if not os.path.isfile(skymapName):
        skyPatch(opts.RAmin, opts.RAmax, opts.nRA,
                 opts.Decmin, opts.Decmax, opts.nDec,
                 skymapName)

    patches = np.load(skymapName)

else:
    r = [(opts.RAmin, opts.RAmax, opts.Decmin, opts.Decmax)]
    patches = np.rec.fromrecords(
        r, names=['RAmin', 'RAmax', 'Decmin', 'Decmax'])


print('patches', patches)
df_tot = pd.DataFrame()
if opts.fieldType == 'WFD':
    for patch in patches:
        proc = procObsPixels(opts.outDir,
                             opts.dbDir, opts.dbName, opts.dbExtens,
                             opts.remove_dithering, opts.fieldType,
                             patch['RAmin'], patch['RAmax'],
                             patch['Decmin'], patch['Decmax'], opts.nside,
                             opts.nprocs, saveData=opts.saveData,
                             fieldName=opts.fieldName,
                             nclusters=opts.nclusters,
                             radius=opts.radius,
                             project_FP=opts.project_FP,
                             VRO_FP=opts.VRO_FP,
                             telrot=opts.telrot)
        res = proc()
        df_tot = pd.concat((df_tot, res))

if opts.fieldType == 'DD':
    fieldName = opts.fieldName.split(',')
    patch = patches[0]
    proc = procObsPixels(opts.outDir,
                         opts.dbDir, opts.dbName, opts.dbExtens,
                         opts.remove_dithering, opts.fieldType,
                         patch['RAmin'], patch['RAmax'],
                         patch['Decmin'], patch['Decmax'], opts.nside,
                         opts.nprocs, saveData=opts.saveData,
                         fieldName=opts.fieldName,
                         radius=opts.radius,
                         project_FP=opts.project_FP,
                         VRO_FP=opts.VRO_FP, telrot=opts.telrot)
    proc()
