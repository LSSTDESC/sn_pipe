#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:54:41 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_tools.sn_utils import multiproc
import os
import pandas as pd

def convert_multi(list_os, params, j=0, output_q=None):
    """
    Function to concert a list of OS 

    Parameters
    ----------
    list_os : pandas df
        list of os to convert.
    params : dict
        parameters.
    j : int, optional
        internal int for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        where to put the results. The default is None.

    Returns
    -------
    int
        output tag.

    """
    
    inputDir = params['inputDir']
    outputDir = params['outputDir']
    
    scr = 'python run_scripts/utils/convert_db_to_npy.py'
    for i, row in list_os.iterrows():
        
        cmd = scr
        cmd += ' --dbName={}'.format(row['dbName'].split('.db')[0])
        cmd += ' --inputDir={}'.format(inputDir)
        cmd += ' --outputDir={}'.format(outputDir)
        
        os.system(cmd)
        #print(cmd)
        
        
    if output_q is not None:
        return output_q.put({j: j})
    else:
        return j


desc = 'script to convert multiple db files to npy ones'
parser = OptionParser(description=desc)

parser.add_option("--list_OS", type=str, default='list_OS.csv',
                  help="list of OS to convert [%default]")    
parser.add_option("--inputDir", type=str, 
                  default='/sps/lsst/groups/cadence/LSST_SN_PhG/cadence_db/fbs_5.3/db',
                  help="input dir of files to convert [%default]") 
parser.add_option("--outputDir", type=str, 
                  default='/sps/lsst/groups/cadence/LSST_SN_PhG/cadence_db/fbs_5.3/npy',
                  help="output dir of converted files [%default]")
parser.add_option("--nproc", type=int, default=8,
                  help="nproc for multiprocessing [%default]")

opts, args = parser.parse_args()

params = {}
params['inputDir'] = opts.inputDir
params['outputDir'] = opts.outputDir

df_list = pd.read_csv(opts.list_OS,comment='#')

multiproc(df_list,params,convert_multi,opts.nproc)