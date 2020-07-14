import os
import numpy as np
from optparse import OptionParser
import glob
import pandas as pd
"""
def batch(dbDir, dbName, scriptref, nproc, season, diffflux, outDir):
    cwd = os.getcwd()
    dirScript = cwd + "/scripts"

    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)

    id = '{}_{}_{}'.format(dName, season, diffflux)
    name_id = 'metric_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'

    qsub = 'qsub -P P_lsst -l sps=1,ct=10:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
        log, nproc)
    # qsub = "qsub -P P_lsst -l sps=1,ct=05:00:00,h_vmem=16G -j y -o "+ log + " <<EOF"
    scriptName = dirScript+'/'+name_id+'.sh'

    script = open(scriptName, "w")
    script.write(qsub + "\n")
    # script.write("#!/usr/local/bin/bash\n")
    script.write("#!/bin/env bash\n")
    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh CCIN2P3\n")
    script.write("echo 'sourcing done' \n")

    cmd = 'python {} --dbDir {} --dbName {} --nproc {} --season {} --diffflux {} --outDir {}'.format(
        scriptref, dbDir, dbName, nproc, season, diffflux, outDir)
    script.write(cmd+" \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)


def batch_family(dbDir, familyName, arrayDb, scriptref, nproc, diffflux, outDir, x1, color, zmin, zmax):
    cwd = os.getcwd()
    dirScript = cwd + "/scripts"

    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)

    id = '{}_{}'.format(familyName, diffflux)
    name_id = 'simulation_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'

    qsub = 'qsub -P P_lsst -l sps=1,ct=05:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
        log, nproc)
    # qsub = "qsub -P P_lsst -l sps=1,ct=05:00:00,h_vmem=16G -j y -o "+ log + " <<EOF"
    scriptName = dirScript+'/'+name_id+'.sh'

    script = open(scriptName, "w")
    script.write(qsub + "\n")
    # script.write("#!/usr/local/bin/bash\n")
    script.write("#!/bin/env bash\n")
    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh CCIN2P3\n")
    script.write("echo 'sourcing done' \n")

    for dbName in arrayDb['dbName']:
        for season in range(1, 11):
            cmd = 'python {} --dbDir {} --dbName {} --nproc {} --season {} --diffflux {} --outDir {}'.format(
                scriptref, dbDir, dbName, 1, season, diffflux, '{}/{}'.format(outDir, familyName))
            cmd += ' --x1 {} --color {} --zmin {} --zmax {}'.format(
                x1, color, zmin, zmax)
            script.write(cmd+" \n")
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)

"""
"""
dbDir ='/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db/2018-06-WPC'

dbNames=['kraken_2026','kraken_2042','kraken_2035','kraken_2044']
dbNames = ['kraken_2026','kraken_2042','kraken_2035','kraken_2044',
    'colossus_2667','pontus_2489','pontus_2002','mothra_2049','nexus_2097']

for dbName in dbNames:
    batch(dbDir,dbName,'run_metric',8)


dbDir = '/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db'
dbNames = ['alt_sched', 'alt_sched_rolling',
           'rolling_10yrs', 'rolling_mix_10yrs']
dbNames += ['kraken_2026', 'kraken_2042', 'kraken_2035', 'kraken_2044',
            'colossus_2667', 'pontus_2489', 'pontus_2002', 'mothra_2049', 'nexus_2097']
dbNames += ['baseline_1exp_nopairs_10yrs', 'baseline_1exp_pairsame_10yrs', 'baseline_1exp_pairsmix_10yrs', 'baseline_2exp_pairsame_10yrs',
            'baseline_2exp_pairsmix_10yrs', 'ddf_0.23deg_1exp_pairsmix_10yrs', 'ddf_0.70deg_1exp_pairsmix_10yrs',
            'ddf_pn_0.23deg_1exp_pairsmix_10yrs', 'ddf_pn_0.70deg_1exp_pairsmix_10yrs', 'exptime_1exp_pairsmix_10yrs', 'baseline10yrs',
            'big_sky10yrs', 'big_sky_nouiy10yrs', 'gp_heavy10yrs', 'newA10yrs', 'newB10yrs', 'roll_mod2_mixed_10yrs',
            'roll_mod3_mixed_10yrs', 'roll_mod6_mixed_10yrs', 'simple_roll_mod10_mixed_10yrs', 'simple_roll_mod2_mixed_10yrs',
            'simple_roll_mod3_mixed_10yrs', 'simple_roll_mod5_mixed_10yrs', 'twilight_1s10yrs',
            'altsched_1exp_pairsmix_10yrs', 'rotator_1exp_pairsmix_10yrs', 'hyak_baseline_1exp_nopairs_10yrs',
            'hyak_baseline_1exp_pairsame_10yrs']

# dbNames = ['alt_sched','alt_sched_rolling','rolling_10yrs','rolling_mix_10yrs','kraken_2026','kraken_2042']
dbNames = ['alt_sched', 'alt_sched_rolling', 'kraken_2026']


diffflux = 0
outDir = '/sps/lsst/users/gris/Output_Simu_pipeline_{}'.format(diffflux)

for dbName in dbNames:
    for season in range(1,11):
        batch(dbDir,dbName,'run_scripts/run_simulation_fromnpy.py',8,season,diffflux,outDir)


toprocess = np.loadtxt('for_batch/OpsimDB.txt',
                       dtype={'names': ('family', 'dbName'), 'formats': ('U11', 'U36')})

print(toprocess)

x1 = -2.0
color = 0.2
zmin = 0.0
zmax = 0.95

for family in np.unique(toprocess['family']):
    idx = toprocess['family'] == family
    sel = toprocess[idx]
    batch_family(dbDir, family, sel, 'run_scripts/run_simulation_fromnpy.py',
                 8, diffflux, outDir, x1, color, zmin, zmax)

"""


def go_for_batch(toproc, splitSky,
                 dbDir, dbExtens, outDir,
                 nodither, nside, fieldType,
                 pixelmap_dir, npixels,
                 x1Type, x1min, x1max, x1step,
                 colorType, colormin, colormax, colorstep,
                 zType, zmin, zmax, zstep,
                 daymaxType, daymaxstep):
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
    nodither: bool
      to remove the dithering (useful for dedicated DD studies)
    nside: int
      healpix nside parameter
    fieldType: str
      type of field to process (DD, WFD, Fakes)
    pixelmap_dir: str
      directory where pixel maps (ie matched pixel positions and observations) are located
    npixels: int
      number of pixels to process
    x1Type: str
        x1 type for simulation (unique, uniform, random)
    x1min: float
        x1 min for simulation
    x1max: float
        x1 max for simulation
    x1step: float
        x1 step for simulation  (type: uniform)
   colorType: str
        color type for simulation (unique, uniform, random)
    colormin: float
        color min for simulation
    colormax: float
        color max for simulation
    colorstep: float
        color step for simulation  (type: uniform)
    zType: str
        z type for simulation (unique, uniform, random)
    zmin: float
        z min for simulation
    zmax: float
        z max for simulation
    zstep: float
        z step for simulation  (type: uniform)
    daymaxType: str
        daymax type for simulation (unique, uniform, random)
    daymaxstep: float
        daymax step for simulation (type: uniform)
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
            batchclass(dbName, dbDir, dbExtens, 'run_scripts/simulation/run_simulation',
                       outDir, 8, toproc,
                       nodither, nside, fieldType, RA_min, RA_max,
                       -1.0, -1.0,
                       pixelmap_dir, npixels, npixels,
                       x1Type, x1min, x1max, x1step,
                       colorType, colormin, colormax, colorstep,
                       zType, zmin, zmax, zstep,
                       daymaxType, daymaxstep)

    else:
        # second case: there are pixelmaps available -> run on them
        # first: get the skymap
        fileSky = glob.glob('{}/skypatch*.npy'.format(pixelmap_dir))
        skyMap = np.load(fileSky[0])

        print(skyMap)
        # get the total number of pixels in this skyMap

        # get the total number of pixels - this is requested if npixels >= 0 and npixels!=-1
        # npixels=-1 means processing all pixels

        npixels_tot = 0
        if npixels > 0:
            for val in skyMap:
                search_path = '{}/{}/{}_{}_nside_{}_{}_{}_{}_{}.npy'.format(
                    pixelmap_dir, dbName, dbName, fieldType, nside, val['RAmin'], val['RAmax'], val['Decmin'], val['Decmax'])
                ffi = glob.glob(search_path)
                tab = np.load(ffi[0])
                # print(len(np.unique(tab['healpixID'])))
                npixels_tot += len(np.unique(tab['healpixID']))

        # print(npixels_tot)

        # now redo the loop and run batches
        for val in skyMap:
            # get the number of pixels for this map
            search_path = '{}/{}/{}_{}_nside_{}_{}_{}_{}_{}.npy'.format(
                pixelmap_dir, dbName, dbName, fieldType, nside, val['RAmin'], val['RAmax'], val['Decmin'], val['Decmax'])
            ffi = glob.glob(search_path)
            tab = np.load(ffi[0])
            npixels_map = len(np.unique(tab['healpixID']))

            # print('pixel_map',val['RAmin'],val['RAmax'],npixels_map)
            npixel_proc = npixels
            if npixels > 0:
                num = float(npixels*npixels_map)/float(npixels_tot)
                npixel_proc = int(round(num))
                # print('hoio',npixel_proc,num)
            batchclass(dbName, dbDir, dbExtens, 'run_scripts/simulation/run_simulation',
                       outDir, 8, toproc,
                       nodither, nside, fieldType, val['RAmin'], val['RAmax'],
                       val['Decmin'], val['Decmax'],
                       pixelmap_dir=pixelmap_dir, npixels=npixel_proc,
                       npixels_tot=npixels,
                       x1Type=x1Type, x1min=x1min, x1max=x1max, x1step=x1step,
                       colorType=colorType, colormin=colormin, colormax=colormax, colorstep=colorstep,
                       zType=zType, zmin=zmin, zmax=zmax, zstep=zstep,
                       daymaxType=daymaxType, daymaxstep=daymaxstep)


class batchclass:
    def __init__(self, dbName, dbDir, dbExtens, scriptref, outDir, nproccomp,
                 toprocess, nodither, nside,
                 fieldType='WFD',
                 RA_min=0.0, RA_max=360.0,
                 Dec_min=-1.0, Dec_max=-1.0,
                 pixelmap_dir='', npixels=0, npixels_tot=0,
                 x1Type='unique', x1min=-2.0, x1max=2.0, x1step=0.1,
                 colorType='unique', colormin=0.2, colormax=0.3, colorstep=0.05,
                 zType='uniform', zmin=0.01, zmax=1.0, zstep=0.1,
                 daymaxType='unique', daymaxstep=1):
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
        pixelmap_dir: str, opt
          location directory of pixelmaps (default: '')
        npixels: int, opt
          number of pixels to process (default: 0)
        npixels_tot: int, opt
          number of pixels initially to process (default: 0)
           x1Type: str
        x1 type for simulation (unique, uniform, random)
    x1min: float
        x1 min for simulation
    x1max: float
        x1 max for simulation
    x1step: float
        x1 step for simulation  (type: uniform)
   colorType: str
        color type for simulation (unique, uniform, random)
    colormin: float
        color min for simulation
    colormax: float
        color max for simulation
    colorstep: float
        color step for simulation  (type: uniform)
    zType: str
        z type for simulation (unique, uniform, random)
    zmin: float
        z min for simulation
    zmax: float
        z max for simulation
    zstep: float
        z step for simulation  (type: uniform)
    daymaxType: str
        daymax type for simulation (unique, uniform, random)
    daymaxstep: float
        daymax step for simulation (type: uniform)

        """

        self.dbName = dbName
        self.dbDir = dbDir
        self.dbExtens = dbExtens
        self.scriptref = scriptref
        self.outDir = outDir
        self.nproccomp = nproccomp
        self.toprocess = toprocess
        self.nodither = nodither
        self.RA_min = RA_min
        self.RA_max = RA_max
        self.Dec_min = Dec_min
        self.Dec_max = Dec_max
        self.nside = nside
        self.fieldType = fieldType
        self.pixelmap_dir = pixelmap_dir
        self.npixels = npixels
        self.npixels_tot = npixels_tot
        self.x1Type = x1Type
        self.x1min = x1min
        self.x1max = x1max
        self.x1step = x1step
        self.colorType = colorType
        self.colormin = colormin
        self.colormax = colormax
        self.colorstep = colorstep
        self.zType = zType
        self.zmin = zmin
        self.zmax = zmax
        self.zstep = zstep
        self.daymaxType = daymaxType
        self.daymaxstep = daymaxstep

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
            self.dbName, self.nside, self.fieldType, 'simulation',
            self.nodither, self.RA_min, self.RA_max, self.Dec_min, self.Dec_max)
        if self.pixelmap_dir != '':
            id += '_frompixels_{}_{}'.format(self.npixels, self.npixels_tot)

        name_id = 'simulation_{}'.format(id)
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

        cmd_ = self.batch_cmd(proc,name_id)
        script.write(cmd_+" \n")

        script.write("EOF" + "\n")
        script.close()
        os.system("sh "+scriptName)

    def batch_cmd(self, proc,name_id):
        """
        Method for the batch command

        Parameters
        ----------------
        proc: numpy array
          data to process

        """

        cmd = 'python {}.py --dbDir {} --dbName {} --dbExtens {}'.format(
            self.scriptref, self.dbDir, proc['dbName'], self.dbExtens)
        """
        cmd += ' --nproc {} --nside {} --simuType {}'.format(
            proc['nproc'], proc['nside'], proc['simuType'])
        """
        cmd += ' --nproc {} --nside {}'.format(
            proc['nproc'], proc['nside'])
        cmd += ' --outDir {}'.format(self.outDir)
        cmd += ' --fieldType {}'.format(self.fieldType)
        cmd += ' --coadd {}'.format(proc['coadd'])
        if self.nodither != '':
            cmd += ' --nodither {}'.format(self.nodither)

        cmd += ' --RAmin {}'.format(self.RA_min)
        cmd += ' --RAmax {}'.format(self.RA_max)
        cmd += ' --Decmin {}'.format(self.Dec_min)
        cmd += ' --Decmax {}'.format(self.Dec_max)

        if self.pixelmap_dir != '':
            cmd += ' --pixelmap_dir {}'.format(self.pixelmap_dir)
            cmd += ' --npixels {}'.format(self.npixels)

        cmd += ' --x1Type {}'.format(self.x1Type)
        cmd += ' --x1min {}'.format(self.x1min)
        cmd += ' --x1max {}'.format(self.x1max)
        cmd += ' --x1step {}'.format(self.x1step)

        cmd += ' --colorType {}'.format(self.colorType)
        cmd += ' --colormin {}'.format(self.colormin)
        cmd += ' --colormax {}'.format(self.colormax)
        cmd += ' --colorstep {}'.format(self.colorstep)

        cmd += ' --zType {}'.format(self.zType)
        cmd += ' --zmin {}'.format(self.zmin)
        cmd += ' --zmax {}'.format(self.zmax)
        cmd += ' --zstep {}'.format(self.zstep)

        cmd += ' --daymaxType {}'.format(self.daymaxType)
        cmd += ' --daymaxstep {}'.format(self.daymaxstep)

        cmd += ' --prodid {}'.format(name_id)
        return cmd


parser = OptionParser()

parser.add_option("--dbList", type="str", default='WFD.txt',
                  help="dbList to process  [%default]")
parser.add_option("--dbDir", type="str", default='/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.4/db',
                  help="db dir [%default]")
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
parser.add_option("--nodither", type="str", default='',
                  help="db extension [%default]")
parser.add_option("--splitSky", type="int", default=0,
                  help="db extension [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="nside healpix parameter[%default]")
parser.add_option("--fieldType", type=str, default='WFD',
                  help="field type[%default]")
parser.add_option("--pixelmap_dir", type=str, default='',
                  help="dir where to find pixel maps[%default]")
parser.add_option("--npixels", type=int, default=0,
                  help="number of pixels to process[%default]")
parser.add_option("--outDir", type=str, default='/sps/lsst/users/gris/Simulations',
                  help="output directory[%default]")
parser.add_option("--x1Type", type=str, default='unique',
                  help="x1 type - unique, uniform, random[%default]")
parser.add_option("--x1min", type=float, default=-2.0,
                  help="x1 min value [%default]")
parser.add_option("--x1max", type=float, default=2.0,
                  help="x1 max [%default]")
parser.add_option("--x1step", type=float, default=0.1,
                  help="x1 step - type = uniform only[%default]")

parser.add_option("--colorType", type=str, default='unique',
                  help="color type - unique, uniform, random[%default]")
parser.add_option("--colormin", type=float, default=0.2,
                  help="color min value [%default]")
parser.add_option("--colormax", type=float, default=0.3,
                  help="color max [%default]")
parser.add_option("--colorstep", type=float, default=0.05,
                  help="color step - type = uniform only[%default]")

parser.add_option("--zType", type=str, default='unique',
                  help="z type - unique, uniform, random[%default]")
parser.add_option("--zmin", type=float, default=0.1,
                  help="z min value [%default]")
parser.add_option("--zmax", type=float, default=1.0,
                  help="z max [%default]")
parser.add_option("--zstep", type=float, default=0.1,
                  help="z step - type = uniform only[%default]")

parser.add_option("--daymaxType", type=str, default='unique',
                  help="daymax type - unique, uniform, random[%default]")
parser.add_option("--daymaxstep", type=float, default=1.,
                  help="daymax step - type = uniform only[%default]")

opts, args = parser.parse_args()

print('Start processing...')

dbList = opts.dbList
dbDir = opts.dbDir
dbExtens = opts.dbExtens
outDir = opts.outDir
nodither = opts.nodither
splitSky = opts.splitSky
nside = opts.nside
fieldType = opts.fieldType
pixelmap_dir = opts.pixelmap_dir
npixels = opts.npixels

x1Type = opts.x1Type
x1min = opts.x1min
x1max = opts.x1max
x1step = opts.x1step

colorType = opts.colorType
colormin = opts.colormin
colormax = opts.colormax
colorstep = opts.colorstep

zType = opts.zType
zmin = opts.zmin
zmax = opts.zmax
zstep = opts.zstep

daymaxType = opts.daymaxType
daymaxstep = opts.daymaxstep
"""
toprocess = np.genfromtxt(dbList, dtype=None, names=[
    'dbName', 'simuType', 'nside', 'coadd', 'fieldType', 'nproc'])

print('there', toprocess, type(toprocess), toprocess.size)

if toprocess.size == 1:
    toprocess = np.array([toprocess])
"""
toprocess = pd.read_csv(dbList, comment='#')
print('there', toprocess, type(toprocess), toprocess.size)

for index, proc in toprocess.iterrows():
    myproc = go_for_batch(proc, splitSky,
                          dbDir, dbExtens, outDir,
                          nodither, nside, fieldType,
                          pixelmap_dir, npixels,
                          x1Type, x1min, x1max, x1step,
                          colorType, colormin, colormax, colorstep,
                          zType, zmin, zmax, zstep,
                          daymaxType, daymaxstep)
