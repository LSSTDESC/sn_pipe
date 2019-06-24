import h5py
import numpy as np
import glob
from astropy.table import Table, vstack, Column
import matplotlib.pyplot as plt
import os
import multiprocessing
from sn_tools.sn_utils import DiffFlux

dirFiles = '/sps/lsst/users/gris/Output_Simu_pipeline_1'
dbName = 'alt_sched'
season = 1
nproc = 8
#dirFiles = 'Output_Simu'
#dbName = 'Test'

metaFiles=glob.glob('{}/Simu_{}*.hdf5'.format(dirFiles,dbName))
metaFiles = ['/sps/lsst/users/gris/Output_Simu_pipeline_0/Simu_alt_sched_0_5.hdf5']
metaFiles = []

for io in range(nproc):
    metaFiles.append('{}/Simu_{}_{}_{}.hdf5'.format(dirFiles,dbName,io,season))

for metaFile in metaFiles:
    DiffFlux(metaFile,dirFiles,outDir='/sps/lsst/users/gris/Output_Simu_pipeline_diffflux')
    break
