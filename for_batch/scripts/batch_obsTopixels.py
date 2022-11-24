import os
import numpy as np
from optparse import OptionParser
import pandas as pd
import copy
from sn_tools.sn_batchutils import BatchIt


def batch_allDDF(params, toprocess,tag):
    """
    Function to generate a single script for a set of DDF db
    and launch the batch

    Parameters
    ----------------
    params: dict
      dict of parameters
    toprocess: pandas df
      list of db and params to process
    tag: int
      additional info to tag the script
    """
    pp = copy.deepcopy(params)
    processName = 'obsTopixels_allDD_{}_{}_{}_{}'.format(pp['nside'],pp['project_FP'],pp['VRO_FP'],tag)
    mybatch = BatchIt(processName=processName)
    outDir_main = params['outDir']
    pp['outDir'] = '{}_{}_{}_{}'.format(outDir_main,pp['nside'],pp['project_FP'],pp['VRO_FP'])
    for io, val in toprocess.iterrows():
        for vv in vvars:
            pp[vv] = val[vv]
      
        mybatch.add_batch(scriptref, pp)

    mybatch.go_batch()


def batch_indiv(params, toprocess, mode):
    """
    Method to generate a script and launch it per db

    Parameters
    ----------------
    params: dict
       dict of parameters
    toprocess: pandas df
      list of db and params to process
    mode: str
      mode of running (batch or interactive)

    """

    for io, val in toprocess.iterrows():
        bid = '{}_{}_{}'.format(val['dbName'], val['nside'], val['fieldType'])
        if mode == 'batch':
            if val['fieldType'] == 'DD':
                params['nclusters'] = nclusters
                bbid = '{}_{}'.format(bid,opts.DDFs.replace(',','_'))
                bbatch(scriptref,params,vvars,val,bbid)
            else:
                bbatch(scriptref,params,vvars,val,bid)
        else:
            scrName = 'test_{}.sh'.format(bid)
            for vv in vvars:
                params[vv] = val[vv]
            make_scr(scriptref,params,scrName)
            os.system('chmod +x {}'.format(scrName))
        


def bbatch(scriptref,params,vvars,val,bid):
    """
    Function to generate batch script and launc it

    Parameters
    ----------------
    scriptref: str
      name of the script to run in batch
    params: dict
      dict of parameters
    vvars: list(str)
      additionnal parameter names
    val: dict
      additionnal parameter values
    bid: str
      batch id

    """
    processName =  'obsTopixels_{}'.format(bid)
    mybatch = BatchIt(processName=processName)
    for vv in vvars:
        params[vv] = val[vv]
    mybatch.add_batch(scriptref,params)
    mybatch.go_batch()

def make_scr(scriptref,params,scrName):
    script = open(scrName, "w")

    script.write('#!/bin/bash \n')
    cmd = 'python {}'.format(scriptref)
    
    for key,vals in params.items():
        cmd += ' --{} {}'.format(key,vals)

    script.write(cmd+'\n')


    cmd = "srun -p htc_interactive --ntasks-per-core 8 {}".format(scrName)

    print('executing',cmd)
    os.system(cmd)


parser = OptionParser()

parser.add_option("--dbList", type="str", default='input/obsTopixels/List_Db_DD.csv',
                  help="dbList to process  [%default]")
parser.add_option("--nodither", type="str", default='',
                  help="to remove dithering [%default]")
parser.add_option("--outDir", type="str",
                  default='/sps/lsst/users/gris/ObsPixelized',
                  help="output directory[%default]")
parser.add_option("--nclusters", type=int,
                  default=6,
                  help="number of clusters - for DD only[%default]")
parser.add_option("--RAmin", type=float, default=0.,
                  help="RA min for obs area - [%default]")
parser.add_option("--RAmax", type=float, default=360.,
                  help="RA max for obs area - [%default]")
parser.add_option("--nRA", type=int, default=10,
                  help="number of RA patches - [%default]")
parser.add_option("--Decmin", type=float, default=-1.0,
                  help="Dec min for obs area - [%default]")
parser.add_option("--Decmax", type=float, default=-1.0,
                  help="Dec max for obs area - [%default]")
parser.add_option("--nDec", type=int, default=1,
                  help="number of Dec patches - [%default]")
parser.add_option("--radius", type=float, default=4.,
                  help="radius for pixel arounf center [DD only] - [%default]")
parser.add_option("--DDFs", type=str, default='COSMOS,XMM-LSS,ELAISS1,CDFS,EDFSa,EDFSb',
                  help="list of DDF ro consider - [%default]")
parser.add_option("--mode", type=str, default='batch',
                  help="mode to run (batch/interactive) - [%default]")
parser.add_option("--runType", type=str, default='allDDF',
                  help="type of run - [%default]")
parser.add_option("--VRO_FP", type=str, default='circular',
                  help="VRO Focal Plane (circle or realistic) [%default]")
parser.add_option("--project_FP", type=str, default='gnomonic',
                  help="Focal Plane projection (gnomonic or hp_query) [%default]")
parser.add_option("--nside", type=int, default=128,
                  help="nside [%default]")
parser.add_option("--n_per_batch", type=int, default=5,
                  help="number of OS per batch [%default]")

opts, args = parser.parse_args()

print('Start processing...', opts)

dbList = opts.dbList
outDir = opts.outDir
RAmin = opts.RAmin
RAmax = opts.RAmax
Decmin = opts.Decmin
Decmax = opts.Decmax
nRA = opts.nRA
nDec = opts.nDec
radius = opts.radius
mode = opts.mode
nclusters = opts.nclusters
runType = opts.runType
VRO_FP = opts.VRO_FP
project_FP = opts.project_FP
nside = opts.nside
n_per_batch = opts.n_per_batch

toprocess = pd.read_csv(dbList, comment='#')

# print('there', toprocess)
scriptref = 'run_scripts/obs_pixelize/run_obs_to_pixels.py'
fieldNames = []
#fieldDD = ['COSMOS', 'CDFS', 'ELAIS', 'XMM-LSS', 'ADFS1', 'ADFS2']
fieldDD = opts.DDFs.split(',')

params = {}
params['outDir'] = outDir
params['RAmin'] = RAmin
params['RAmax'] = RAmax
params['nRA'] = nRA
params['Decmin'] = Decmin
params['Decmax'] = Decmax
params['nDec'] = nDec
params['radius'] = radius
params['saveData'] = 1
params['fieldName'] = opts.DDFs
params['VRO_FP'] = VRO_FP
params['project_FP'] = opts.project_FP
params['nside'] = nside

vvars = ['dbDir', 'dbName', 'dbExtens','nproc','fieldType','simuType']

#make a big and unique file for DD
if runType == 'allDDF':
    batch_allDDF(params,toprocess)
else:
    if runType == 'splitDDF':
        print(len(toprocess))
        nn = int(len(toprocess)/n_per_batch)
        dfs = np.array_split(toprocess, nn)
        for io,vv in enumerate(dfs):
            batch_allDDF(params,vv,io)
    else:
        batch_indiv(params,toprocess,mode)

    
    
