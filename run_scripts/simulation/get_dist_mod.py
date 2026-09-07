#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:01:06 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from sn_tools.sn_io import checkDir
from sn_analysis.sn_selection import selection_criteria
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(master_dir,fDir,dbName,runType,sellist):
    
    dirName = '{}/{}/{}/{}'.format(master_dir,fDir,dbName,runType)
    
    pathName = '{}/SN_*.hdf5'.format(dirName)
    
    fis = glob.glob(pathName)
    
    print('allo',pathName)
    df = pd.DataFrame()
    
    for fi in fis:
        dfa = pd.read_hdf(fi)
        df = pd.concat((df,dfa))
        
    print('before',len(df))
    df = complete_data(df,sellist)
    
    print('after',len(df))    
    
    return df
    
def grab_data(z,pp,sellist):
    
    fDir='lc{}_{}_{}_{}'.format(z,pp['config'],pp['config_fit'],pp['config_coadd'])


    df = load_data(pp['master_dir'],fDir,pp['dbName'],pp['runType'],sellist)
    
    return df
    
def complete_data(df,sellist=[]):
    
    
    from sn_analysis.sn_tools import complete_df
    from sn_analysis.sn_selection import select
 
    df['sigma_color'] = np.sqrt(df['Cov_colorcolor'])
 
    df = complete_df(df)
 
    if sellist:
        df = pd.DataFrame(select(df,list_sel=sellist))
 
    for vv in ['x1','color']:
        vdiff = '{}_fit'.format(vv)
        vsigma = 'sigma_{}'.format(vv)
        df['pull_{}'.format(vv)] = (df[vv]-df[vdiff])/df[vsigma]
        df['diff_{}'.format(vv)] = (df[vv]-df[vdiff])
    
    return df

def get_stat(grp,cols=['diff_x1','diff_color','mu','mu_exp']):
    
    print(grp.columns,grp.name)
    
    rmean = grp[cols].mean().to_list()
    rstd = grp[cols].std()
    
    print(type(rmean))
    

parser = OptionParser(description='Script to compare LCs on a large scale')

parser.add_option('--master_dir', type=str, 
                  default='../prod_lcnew',
                  help='master dir for data [%default]')
parser.add_option('--config', type=str, 
                  default='confa',
                  help='config to process [%default]')
parser.add_option('--dbName', type=str, default='baseline_v5.3.0_10yrs',
                  help='OS to process [%default]')
parser.add_option('--runType', type=str, default='DDF_spectroz',
                  help='run type [%default]')
parser.add_option('--configs', type=str, default='confa,conff',
                  help='configs [%default]')
parser.add_option('--config_fit', type=str, default='fit',
                  help='fit config [%default]')
parser.add_option('--config_coadd', type=str, default='coadd',
                  help='coadd config [%default]')

opts, args = parser.parse_args()

pp = vars(opts)
sellist = selection_criteria()['G10_JLA']

zmin = 0.1
zmax = 0.1
zstep = 0.01

zvals = np.arange(zmin,zmax+zstep,zstep)

for z in zvals:
    if z <= zmax:
        df = grab_data(z, pp, sellist)

res = df.groupby(['z']).apply(lambda x: get_stat(x),include_groups=False).reset_index()