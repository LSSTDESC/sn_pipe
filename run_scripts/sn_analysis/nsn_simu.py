#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 13:44:13 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import glob
from sn_tools.sn_io import checkDir
from sn_tools.sn_utils import multiproc

def load_count(dbDir,dbName,runType,nproc=8):
    """
    Function to load files and count using multiprocessing

    Parameters
    ----------
    dbDir : str
        OS dir.
    dbName : str
        OS name.
    runType : str
        run type.
    nproc : int, optional
        number of procs to use. The default is 8.

    Returns
    -------
    pandas df
        two columns: dbName, nsn_simu.

    """
    
    dName = '{}/{}/{}'.format(dbDir,dbName,runType)
    
    fis = glob.glob('{}/*.hdf5'.format(dName))

    params = {}
    df = multiproc(fis,params,process,nproc)
        
    dd = {}
    dd['dbName'] = [dbName]
    dd['nsn_simu'] = df['nsn_simu'].sum()
    
    return pd.DataFrame.from_dict(dd)

def process(fis,params,j=0,output_q=None):
    """
    Function to process a set of files

    Parameters
    ----------
    fis : list(str)
        List of files to process.
    params : dict
        parameters.
    j : int, optional
        internal int for multiproc. The default is 0.
    output_q : multiprocessing queue, optional
        where to put the results. The default is None.

    Returns
    -------
    pandas df
        output data.

    """
    
    nsn_simu = 0
    
    for fi in fis:
        df = pd.read_hdf(fi)
        nsn_simu += len(df)
     
    dd = {}
    dd['process'] = [j]
    dd['nsn_simu'] = nsn_simu
    
    df = pd.DataFrame.from_dict(dd)
    
    if output_q is not None:
        return output_q.put({j: df})
    else:
        return df
    
    
    

parser = OptionParser(
    description='Script to estimate the total number of simulated SNe Ia')
parser.add_option('--dbDir', type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass',
                  help='OS location dir [%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v5.0.0_10yrs',
                  help='OS to process [%default]')
parser.add_option('--runType', type=str,
                  default='WFD_spectroz_nosat',
                  help='type of run [%default]')
parser.add_option('--outDir', type=str,
                  default='../nsn_simu_prod',
                  help='outDir [%default]')
parser.add_option('--outName', type=str,
                  default='nsn_simu_WFD_baseline_v5.0.0_10yrs.csv',
                  help='output file name (csv) [%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='number of procs for multiprocessing [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
runType = opts.runType
outDir = opts.outDir
outName = opts.outName
nproc = opts.nproc


#load and count
df = load_count(dbDir,dbName,runType,nproc)

#check output dir
checkDir(outDir)

#save data as csv files
fName = '{}/{}'.format(outDir,outName)
df.to_csv(fName,index=False)