#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 14:23:04 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import numpy as np
from sn_analysis.sn_tools import load_data, complete_df, pull_it
from sn_analysis.sn_fit_tools import fit_hist
from sn_tools.sn_io import checkDir

def fit_pulls(grp,colnames=['x1','color','mb','daymax']):
    """
    Function to fit puuls

    Parameters
    ----------
    grp : pandas df
        Data to process.
    colnames : list(str), optional
        List of var for pull estimation. 
        The default is ['x1','color','mb','daymax'].

    Returns
    -------
    res : pandas df
        result.

    """
    
    res = pd.DataFrame()
    for ccol in colnames:
        pullvar = 'pull_{}'.format(ccol)
        idx = np.abs(grp[pullvar]) <= 5.
        sel = grp[idx]
        if len(sel)< 20:
            continue
        resb = fit_hist(sel,pullvar)
        res = pd.concat((res,resb))
    
    return res
    
    
    
    

parser = OptionParser(description='Script to estimate SN parameter pulls')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_newatm_lc_coadd_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v5.0.0_10yrs',
                  help='OS name [%default]')
parser.add_option('--runType', type=str,
                  default='DDF_spectroz',
                  help='run type [%default]')
parser.add_option('--outDir', type=str,
                  default='../pull_result',
                  help='output directory [%default]')
parser.add_option('--outName', type=str,
                  default='baseline_v5.0.0_10yrs',
                  help='output file name [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
runType = opts.runType
outDir = '{}/{}'.format(opts.outDir,opts.outName)

if outDir != '':
    checkDir(outDir)
# load data

df = load_data(dbDir, dbName, runType)

# complete data
df = complete_df(df)

# estimate pull
df = pull_it(df)

# fit pulls
ccols = ['field','season','healpixID']
res = df.groupby(ccols).apply(lambda x: fit_pulls(x),include_groups=False).reset_index()

fName = '{}/{}.hdf5'.format(outDir,runType)
res.to_hdf(fName,key='pull')
