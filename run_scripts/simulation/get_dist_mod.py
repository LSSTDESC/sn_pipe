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
import operator as op

def load_data(master_dir,fDir,dbName,runType,sellist):
    """
    Function to load data

    Parameters
    ----------
    master_dir : str
        main data dir.
    fDir : str
        file dir.
    dbName : str
        OS name.
    runType : str
        run type.
    sellist : dict
        selection criteria.

    Returns
    -------
    df : pandas df
        Data.

    """
    
    dirName = '{}/{}/{}/{}'.format(master_dir,fDir,dbName,runType)
    
    pathName = '{}/SN_*.hdf5'.format(dirName)
    
    fis = glob.glob(pathName)
    
    df = pd.DataFrame()
    
    for fi in fis:
        dfa = pd.read_hdf(fi)
        df = pd.concat((df,dfa))
        
    df = complete_data(df,sellist)  
    
    return df
    
def grab_data(z,pp,sellist):
    """
    Function to load data (z)

    Parameters
    ----------
    z : float
        redshift value.
    pp : dict
        parameters.
    sellist : dict
        selection criteria.

    Returns
    -------
    df : pandas df
        Data.

    """
    
    fDir='lc{}_{}_{}_{}'.format(z,pp['config'],pp['config_fit'],pp['config_coadd'])

    df = load_data(pp['master_dir'],fDir,pp['dbName'],pp['runType'],sellist)
    
    return df
    
def complete_data(df,sellist=[]):
    """
    Function to complete SN data

    Parameters
    ----------
    df : pandas df
        Data to process.
    sellist : dict, optional
        selection criteria. The default is [].

    Returns
    -------
    df : pandas df
        Resulting df.

    """
    
    
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

def get_stat(grp,cols=['x1','color','mu']):
    """
    Function to get some stat

    Parameters
    ----------
    grp : pandas df
        Data to process.
    cols : list(str), optional
        columns to consider. The default is ['x1','color','mu'].

    Returns
    -------
    df : pandas df
        Result.

    """
    
    
    cols_diff = list(map(lambda it: 'diff_{}'.format(it), cols))
    cols_diff_sigma = list(map(lambda it: 'diff_{}_sigma'.format(it), cols))
    cols_diff_std = list(map(lambda it: 'diff_{}_std'.format(it), cols))
    cols_sigma = list(map(lambda it: 'sigma_{}'.format(it), cols))
    cols_ci = list(map(lambda it: 'ci_{}'.format(it), cols))
    cols_diff_mean = list(map(lambda it: '{}_mean'.format(it), cols_diff))
    cols_diff_med = list(map(lambda it: '{}_med'.format(it), cols_diff))
    
    """
    grp[cols_ci] = 1./grp[cols_sigma]**2
    
    dd = {}
    for i,vv in enumerate(cols_mean):
        dd[vv] = np.sum(grp[cols_diff[i]]*grp[cols_ci[i]])/np.sum(grp[cols_ci[i]])
        dd[cols_diff_sigma[i]] = 1./np.sqrt(np.sum(grp[cols_ci[i]]))
    print(dd)
    """
    
    rmed = grp[cols_diff].median()
    rmean = grp[cols_diff].mean()
    rstd = grp[cols_diff].std()
    
    df = pd.DataFrame([rmed.to_list()],columns=cols_diff_med)
    dfa = pd.DataFrame([rmean.to_list()],columns=cols_diff_mean)
    df = pd.concat((df,dfa),axis=1)
    dfb = pd.DataFrame([rstd.to_list()],columns=cols_diff_std)
    df = pd.concat((df,dfb),axis=1)
    
    df['nsn'] = len(grp)
    
    return df
    
def get_data_z(pp,sellist):
    """
    Function to grab data vs z bins

    Parameters
    ----------
    pp : dict
        parameters.
    sellist : dict
        selection criteria.

    Returns
    -------
    df : pandas df
        processed data.

    """
    
    zmin = 0.01
    zmax = 1.1
    zstep = 0.01
    
    zvals = np.arange(zmin,zmax+zstep,zstep)
    
    df = pd.DataFrame()
    for z in zvals:
        if z <= zmax:
            dfa = grab_data(np.round(z,2), pp, sellist)
            df = pd.concat((df,dfa))
    
    df['config'] = pp['config']
    return df

def process(pp,sellist,outName='comp_distmod.hdf5'):
    """
    Function to process data

    Parameters
    ----------
    pp : dict
        parameters.
    sellist : dict
        selection criteria.
    outName : str, optional
        output file name. The default is 'comp_distmod.hdf5'.

    Returns
    -------
    None.

    """
    
    df = pd.DataFrame()
    configs = ['confa','confb','confc','confd','confe','conff']

    for conf in configs:
        pp['config'] = conf
        dfb = get_data_z(pp,sellist)
        df = pd.concat((df,dfb))

    res = df.groupby(['z','config','season']).apply(lambda x: get_stat(x),include_groups=False).reset_index()

    res.to_hdf(outName,key='distmod')
    
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
parser.add_option('--process', type=int, default=0,
                  help='to force data processing [%default]')
parser.add_option('--config_fit', type=str, default='fit',
                  help='fit config [%default]')
parser.add_option('--config_coadd', type=str, default='coadd',
                  help='coadd config [%default]')

opts, args = parser.parse_args()

pp = vars(opts)
sellist = selection_criteria()['G10_JLA']
sellist.append(('sigma_color',op.le,0.04))

outName='comp_distmod_sigmaC.hdf5'

if pp['process']:
    process(pp,sellist,outName)