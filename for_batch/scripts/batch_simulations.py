import os
import numpy as np
from optparse import OptionParser
import glob
import pandas as pd
import sn_simu_input as simu_input
from sn_tools.sn_io import make_dict_from_config,make_dict_from_optparse


def go_for_batch(toproc,dbDir,dbExtens,run_script,opts):
    """
    Function to prepare and start batches

    Parameters
    ----------------
    toproc: numpy array
      data (dbName, ...) to process
    dbDir: str
      dir where observing strategy files are located
    dbExtens: str
      extension of obs. strategy files (npy or db)
    run_script: str
      script to use for processing
    opts: optparser args
     list of options to apply to the script
    """

    # get the observing strategy name
    #dbName = toproc['dbName'].decode()
    #dbName = toproc['dbName']
    pixelmap_dir = opts.pixelmap_dir
    splitSky = opts.splitSky
    dbName = toproc['dbName']
    Decmin = -1.
    Decmax = -1.

    if pixelmap_dir == '':

        # first case: no pixelmap - run on all the pixels - possibility to split the sky
        n_per_slice = 1
        RAs = [0., 360.]
        if splitSky:
            RAs = np.linspace(0., 360., 11)

        for ira in range(len(RAs)-1):
            RA_min = RAs[ira]
            RA_max = RAs[ira+1]
            batchclass(toproc, dbDir, dbExtens,run_script,RA_min,RA_max,Dec_min,Dec_max,opts)
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
        if opts.npixels > 0:
            for val in skyMap:
                search_path = '{}/{}/{}_{}_nside_{}_{}_{}_{}_{}.npy'.format(
                    pixelmap_dir, dbName, dbName,opts.Observations_fieldtype, opts.Pixelisation_nside, val['RAmin'], val['RAmax'], val['Decmin'], val['Decmax'])
                ffi = glob.glob(search_path)
                tab = np.load(ffi[0])
                # print(len(np.unique(tab['healpixID'])))
                npixels_tot += len(np.unique(tab['healpixID']))

        # print(npixels_tot)

        # now redo the loop and run batches
        for val in skyMap:
            # get the number of pixels for this map
            search_path = '{}/{}/{}_{}_nside_{}_{}_{}_{}_{}.npy'.format(
                pixelmap_dir, dbName, dbName, opts.Observations_fieldtype, opts.Pixelisation_nside,val['RAmin'], val['RAmax'], val['Decmin'], val['Decmax'])
            ffi = glob.glob(search_path)
            tab = np.load(ffi[0],allow_pickle=True)
            npixels_map = len(np.unique(tab['healpixID']))

            print('pixel_map',val['RAmin'],val['RAmax'],npixels_map)
            npixel_proc = opts.npixels
            if opts.npixels > 0:
                num = float(npixels*npixels_map)/float(npixels_tot)
                npixel_proc = int(round(num))
                # print('hoio',npixel_proc,num)
            batchclass(toproc, dbDir, dbExtens, run_script,val['RAmin'], val['RAmax'], val['Decmin'], val['Decmax'],opts,npixels_tot=opts.npixels)
            
class batchclass:
    def __init__(self, toproc,dbDir,dbExtens,run_script,RAmin,RAmax,Decmin,Decmax,opts,npixels_tot):
        """
        class to prepare and launch batches

        Parameters
        ----------------
        opts: parser
          list of parameters to consider

        """
    
        self.dbName = toproc['dbName']
        self.dbDir = dbDir
        self.dbExtens = dbExtens
        self.RAmin = RAmin
        self.RAmax = RAmax
        self.Decmin = Decmin
        self.Decmax = Decmax
        self.npixels_tot = npixels_tot
        self.nproccomp = toproc['nproc']
        

        self.opts = opts
        self.scriptref = run_script
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

        dbName = self.dbName
        
        nside = self.opts.Pixelisation_nside
        fieldType = self.opts.Observations_fieldtype 
        nodither = self.opts.nodither
        

        id = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            dbName, nside, fieldType, 'simulation',
            nodither, self.RAmin, self.RAmax, self.Decmin, self.Decmax)
        if self.opts.pixelmap_dir != '':
            id += '_frompixels_{}_{}'.format(self.opts.npixels, self.npixels_tot)

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
        qsub = 'qsub -P P_lsst -l sps=1,ct=12:00:00,h_vmem=20G -j y -o {} -pe multicores {} <<EOF'.format(
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
            self.scriptref, self.dbDir, self.dbName, self.dbExtens)

        for opt, value in self.opts.__dict__.items():
            if opt not in ['dbList','splitSky','ProductionID','Pixelisation_nside',
                           'Observations_fieldtype','Output_directory','Observations_filename']:
                cmd += ' --{} {}'.format(opt,value)

        cmd += ' --ProductionID {}'.format(name_id)
        cmd += ' --RAmin {}'.format(self.RAmin)
        cmd += ' --RAmax {}'.format(self.RAmax)
        cmd += ' --Decmin {}'.format(self.Decmin)
        cmd += ' --Decmax {}'.format(self.Decmax)
        cmd += ' --nproc {}'.format(proc['nproc'].values[0])
        cmd += ' --Pixelisation_nside {}'.format(proc['nside'].values[0])
        cmd += ' --Observations_fieldtype {}'.format(proc['fieldtype'].values[0])
        cmd += ' --Observations_filename {}/{}.{}'.format(self.dbDir,self.dbName,self.dbExtens)
        cmd += ' --Output_directory {}/{}'.format(opts.Output_directory,self.dbName)
        

        return cmd



# get all possible simulation parameters and put in a dict
path = simu_input.__path__
confDict = make_dict_from_config(path[0],'config_simulation.txt')

parser = OptionParser()

parser.add_option("--dbList", type="str", default='WFD.txt',
                  help="dbList to process  [%default]")
parser.add_option("--dbDir", type="str", default='/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.4/db',
                  help="db dir [%default]")
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
parser.add_option("--splitSky", type="int", default=0,
                  help="db extension [%default]")
parser.add_option("--npixels", type=int, default=0,
                  help="number of pixels to process[%default]")
parser.add_option("--pixelmap_dir", type=str, default='',
                  help="dir where to find pixel maps[%default]")
parser.add_option("--nodither", type="int", default=0,
                  help="to remove dithering [%default]")

# parser for simulation parameters : 'dynamical' generation
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0],vals[1]))
    parser.add_option('--{}'.format(key),help='{} [%default]'.format(vals[2]),default=vv,type=vals[0],metavar='')


opts, args = parser.parse_args()

print('Start processing...')


"""
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
simulator = opts.simulator
"""
"""
toprocess = np.genfromtxt(dbList, dtype=None, names=[
    'dbName', 'simuType', 'nside', 'coadd', 'fieldType', 'nproc'])

print('there', toprocess, type(toprocess), toprocess.size)

if toprocess.size == 1:
    toprocess = np.array([toprocess])
"""
toprocess = pd.read_csv(opts.dbList, comment='#')
print('there', toprocess, type(toprocess), toprocess.size)

for index, proc in toprocess.iterrows():
    print('go here',proc['dbName'])
    myproc = go_for_batch(proc,opts.dbDir,opts.dbExtens,
                          'run_scripts/simulation/run_simulation',opts)
