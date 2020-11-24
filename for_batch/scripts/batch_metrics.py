import os
import numpy as np
from optparse import OptionParser
import glob
import pandas as pd


def go_for_batch(toproc, splitSky,
                 dbDir, dbExtens, outDir, metricName,
                 nodither, nside, fieldType, band,
                 pixelmap_dir, npixels, proxy_level):
    """
    Function to prepare and start batches

    Parameters
    ----------------
    toproc: numpy array
      data (dbName, ...) to process
    splitSky: bool
      to split the batches in sky patches
    dbDir: str
      dir where observing strategy files are located
    dbExtens: str
      extension of obs. strategy files (npy or db)
    outDir: str
      output directory for the produced data
    metricName: str
      name of the metric to run
    nodither: bool
      to remove the dithereing (useful for dedicated DD studies)
    nside: int
      healpix nside parameter
    fieldType: str
      type of field to process (DD, WFD, Fakes)
    band: str
      band to consider (for some metrics like SNR)
    pixelmap_dir: str
      directory where pixel maps (ie matched pixel positions and observations) are located
    npixels: int
      number of pixels to process
    proxy_level: int
      proxy level - for NSN metric only (possible values: 0,1,2)

    """

    # get the observing strategy name
    #dbName = toproc['dbName'].decode()
    dbName = toproc['dbName']

    if pixelmap_dir == '':

        # first case: no pixelmap - run on all the pixels - possibility to split the sky
        n_per_slice = 1
        RAs = [0., 360.]
        if splitSky:
            RAs = np.linspace(0., 360., 11)

        for ira in range(len(RAs)-1):
            RA_min = RAs[ira]
            RA_max = RAs[ira+1]
            batchclass(dbName, dbDir, dbExtens, 'run_scripts/metrics/run_metrics',
                       outDir, 8, 1, metricName, toproc,
                       nodither, nside, fieldType, RA_min, RA_max,
                       -1.0, -1.0, band,
                       pixelmap_dir, npixels, proxy_level, npixels)

    else:
        # second case: there are pixelmaps available -> run on them
        # first: get the skymap
        fileSky = glob.glob('{}/skypatch*.npy'.format(pixelmap_dir))
        skyMap = np.load(fileSky[0],allow_pickle=True)

        print(skyMap)
        # get the total number of pixels in this skyMap

        # get the total number of pixels - this is requested if npixels >= 0 and npixels!=-1
        # npixels=-1 means processing all pixels

        npixels_tot = 0
        if npixels > 0:
            for val in skyMap:
                search_path = '{}/{}/{}_{}_nside_{}_{}_{}_{}_{}_WFD.npy'.format(
                    pixelmap_dir, dbName, dbName, fieldType, nside, val['RAmin'], val['RAmax'], val['Decmin'], val['Decmax'])
                ffi = glob.glob(search_path)
                if len(ffi) == 0:
                    print('potential problem here', search_path)
                tab = np.load(ffi[0],allow_pickle=True)
                # print(len(np.unique(tab['healpixID'])))
                npixels_tot += len(np.unique(tab['healpixID']))

        # print(npixels_tot)

        # now redo the loop and run batches
        for val in skyMap:
            # get the number of pixels for this map
            search_path = '{}/{}/{}_{}_nside_{}_{}_{}_{}_{}_WFD.npy'.format(
                pixelmap_dir, dbName, dbName, fieldType, nside, val['RAmin'], val['RAmax'], val['Decmin'], val['Decmax'])
            ffi = glob.glob(search_path)
            if len(ffi) == 0:
                print('potential problem here', search_path)
            tab = np.load(ffi[0],allow_pickle=True)
            npixels_map = len(np.unique(tab['healpixID']))

            # print('pixel_map',val['RAmin'],val['RAmax'],npixels_map)
            npixel_proc = npixels
            if npixels > 0:
                num = float(npixels*npixels_map)/float(npixels_tot)
                npixel_proc = int(round(num))
                # print('hoio',npixel_proc,num)
            batchclass(dbName, dbDir, dbExtens, 'run_scripts/metrics/run_metrics',
                       outDir, 8, 1, metricName, toproc,
                       nodither, nside, fieldType, val['RAmin'], val['RAmax'],
                       val['Decmin'], val['Decmax'], band=band,
                       pixelmap_dir=pixelmap_dir, npixels=npixel_proc,
                       proxy_level=proxy_level, npixels_tot=npixels)


class batchclass:
    def __init__(self, dbName, dbDir, dbExtens, scriptref, outDir, nproccomp,
                 saveData, metric, toprocess, nodither, nside,
                 fieldType='WFD',
                 RA_min=0.0, RA_max=360.0,
                 Dec_min=-1.0, Dec_max=-1.0, band='',
                 pixelmap_dir='', npixels=0, npixels_tot=0,
                 proxy_level=-1):
        """
        class to prepare and launch batches

        Parameters
        ----------------
        dbName: str
          observing strategy name
        dbDir: str
          location dir of obs. strat. file
        dbExtens: str
          obs. strat. file extension (db or npy)
        scriptref: str
          reference script to use in the batch
        outDir: str
          output directory location
        nproccomp: int
          number of multiproc used
        saveData: bool
          to save the data on disk or not
        metric: str
          name of the metric to run
        toprocess: numpy array
          array of data to process
        nodither: bool
          to remove dither (can be usefull for DD studies)
        nside: int
          healpix nside parameter
        fieldType: str, opt
          type of field to process - DD, WFD, Fakes (default: WFD)
        RA_min: float, opt
          min RA of the area to process (default:0.0)
        RA_max: float, opt
          max RA of the area to process (default: =360.0) 
        Dec_min: float, opt
          min Dec of the area to process (default: -1.0)
        Dec_max: float, opt
          max Dec of the area to process (default: -1.0)
        band: str, opt
          band to process (default: '')
        pixelmap_dir: str, opt
          location directory of pixelmaps (default: '')
        npixels: int, opt
          number of pixels to process (default: 0)
        npixels_tot: int, opt
          number of pixels initially to process (default: 0)
        proxy_level: int, opt
          proxy level for NSN metric (default: -1)

        """

        self.dbName = dbName
        self.dbDir = dbDir
        self.dbExtens = dbExtens
        self.scriptref = scriptref
        self.outDir = outDir
        self.nproccomp = nproccomp
        self.saveData = saveData
        self.metric = metric
        self.toprocess = toprocess
        self.nodither = nodither
        self.RA_min = RA_min
        self.RA_max = RA_max
        self.Dec_min = Dec_min
        self.Dec_max = Dec_max
        self.band = band
        self.nside = nside
        self.fieldType = fieldType
        self.pixelmap_dir = pixelmap_dir
        self.npixels = npixels
        self.npixels_tot = npixels_tot
        self.proxy_level = proxy_level

        dirScript, name_id, log = self.prepareOut()

        self.script(dirScript, name_id, log, toprocess)

    def prepareOut(self):
        """
        Method to prepare for the batch

        directories for scripts and log files are defined here.

        """

        self.cwd = os.getcwd()
        dirScript = self.cwd + "/scripts"

        if not os.path.isdir(dirScript):
            os.makedirs(dirScript)

        dirLog = self.cwd + "/logs"
        if not os.path.isdir(dirLog):
            os.makedirs(dirLog)

        id = '{}_{}_{}_{}{}_{}_{}_{}_{}'.format(
            self.dbName, self.nside, self.fieldType, self.metric,
            self.nodither, self.RA_min, self.RA_max, self.Dec_min, self.Dec_max)
        if self.proxy_level > -1:
            id += '_proxy_level_{}'.format(self.proxy_level)
        if self.pixelmap_dir != '':
            id += '_frompixels_{}_{}'.format(self.npixels, self.npixels_tot)

        name_id = 'metric_{}'.format(id)
        log = dirLog + '/'+name_id+'.log'

        return dirScript, name_id, log

    def script(self, dirScript, name_id, log, proc):
        """
        Method to generate and run the script to be executed

        Parameters
        ----------------
        dirScript: str
          location directory of the script
        name_id: str
          id for the script
        log: str
          location directory for the log files
        proc: numpy array
          data to process

        """
        # qsub command
        qsub = 'qsub -P P_lsst -l sps=1,ct=12:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
            log, self.nproccomp)

        scriptName = dirScript+'/'+name_id+'.sh'

        # fill the script
        script = open(scriptName, "w")
        script.write(qsub + "\n")
        script.write("#!/bin/env bash\n")
        script.write(" cd " + self.cwd + "\n")
        script.write(" echo 'sourcing setups' \n")
        script.write(" source setup_release.sh Linux\n")
        script.write("echo 'sourcing done' \n")

        cmd_ = self.batch_cmd(proc)
        script.write(cmd_+" \n")

        script.write("EOF" + "\n")
        script.close()
        #os.system("sh "+scriptName)

    def batch_cmd(self, proc):
        """
        Method for the batch command

        Parameters
        ----------------
        proc: numpy array
          data to process

        """

        cmd = 'python {}.py --dbDir {} --dbName {} --dbExtens {}'.format(
            self.scriptref, self.dbDir, proc['dbName'], self.dbExtens)
        cmd += ' --nproc {} --nside {} --simuType {}'.format(
            proc['nproc'], proc['nside'], proc['simuType'])
        cmd += ' --outDir {}'.format(self.outDir)
        cmd += ' --fieldType {}'.format(self.fieldType)
        cmd += ' --saveData {}'.format(self.saveData)
        cmd += ' --metric {}'.format(self.metric)
        cmd += ' --coadd {}'.format(proc['coadd'])
        if self.nodither != '':
            cmd += ' --nodither {}'.format(self.nodither)

        cmd += ' --RAmin {}'.format(self.RA_min)
        cmd += ' --RAmax {}'.format(self.RA_max)
        cmd += ' --Decmin {}'.format(self.Dec_min)
        cmd += ' --Decmax {}'.format(self.Dec_max)
        if self.band != '':
            cmd += ' --band {}'.format(self.band)

        if self.pixelmap_dir != '':
            cmd += ' --pixelmap_dir {}'.format(self.pixelmap_dir)
            cmd += ' --npixels {}'.format(self.npixels)
        if self.proxy_level > -1:
            cmd += ' --proxy_level {}'.format(self.proxy_level)

        return cmd


parser = OptionParser()

parser.add_option("--dbList", type="str", default='WFD.txt',
                  help="dbList to process  [%default]")
parser.add_option("--metricName", type="str", default='SNR',
                  help="metric to process  [%default]")
parser.add_option("--dbDir", type="str", default='/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.4/db',
                  help="db dir [%default]")
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
parser.add_option("--nodither", type="str", default='',
                  help="db extension [%default]")
parser.add_option("--splitSky", type="int", default=0,
                  help="db extension [%default]")
parser.add_option("--band", type="str", default='',
                  help="db extension [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="nside healpix parameter[%default]")
parser.add_option("--fieldType", type=str, default='WFD',
                  help="field type[%default]")
parser.add_option("--pixelmap_dir", type=str, default='',
                  help="dir where to find pixel maps[%default]")
parser.add_option("--npixels", type=int, default=0,
                  help="number of pixels to process[%default]")
parser.add_option("--outDir", type=str, default='/sps/lsst/users/gris/MetricOutput_pixels',
                  help="output directory[%default]")
parser.add_option("--proxy_level", type=int, default=-1,
                  help="proxy level - For NSN metric only[%default]")

opts, args = parser.parse_args()

print('Start processing...')

dbList = opts.dbList
metricName = opts.metricName
dbDir = opts.dbDir
band = opts.band
dbExtens = opts.dbExtens

outDir = opts.outDir
nodither = opts.nodither
splitSky = opts.splitSky
nside = opts.nside
fieldType = opts.fieldType
pixelmap_dir = opts.pixelmap_dir
npixels = opts.npixels
proxy_level = opts.proxy_level

# toprocess = np.genfromtxt(dbList, dtype=None, names=[
#                          'dbName', 'simuType', 'nside', 'coadd', 'fieldType', 'nproc'])

toprocess = pd.read_csv(dbList, comment='#')
print('there', toprocess, type(toprocess), toprocess.size)


# if toprocess.size == 1:
#    toprocess= np.array([toprocess])
"""
proc  = batchclass(dbDir, dbExtens, scriptref, outDir, nproccomp,
                 saveData, metric, toprocess, nodither,nside,
                 RA_min=0.0, RA_max=360.0, 
                 Dec_min=-1.0, Dec_max=-1.0,band=''
"""

for index, proc in toprocess.iterrows():
    myproc = go_for_batch(proc, splitSky,
                          dbDir, dbExtens, outDir,
                          metricName, nodither, nside, fieldType,
                          band, pixelmap_dir, npixels, proxy_level)
    # break
