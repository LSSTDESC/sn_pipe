#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:20:05 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import numpy as np
from sn_tools.sn_obs import season
from sn_tools.sn_io import checkDir

def load_data(dbDir,dbNames):
    """
    Function to load the data

    Parameters
    ----------
    dbDir : str
        Data directory.
    dbNames : list(str)
        List of dbs to process.

    Returns
    -------
    data : pandas df
        output data.

    """

    data = pd.DataFrame()
    for dbName in dbNames:
        fName = '{}/{}.hdf5'.format(dbDir, dbName)

        dat_ = pd.read_hdf(fName)
        print(dat_.columns)
        dats = dat_.groupby(['field']).apply(lambda x: get_season(x),
                                             include_groups=False).reset_index()
        
        data = pd.concat((data,dats))

    return data

def get_season(grp):
    """
    Function to estimate the seasons

    Parameters
    ----------
    grp : pandas df
        data to process.

    Returns
    -------
    pandas df
        orig df plus season column.

    """
    
    dats = season(grp.to_records(index=False),mjdCol='night')
    
    return pd.DataFrame.from_records(dats)


def analyze_DDF(df):
    """
    Function to analyze DDF

    Parameters
    ----------
    df : pandas df
        Data to process.

    Returns
    -------
    pandas df
      analysis result

    """
    
    res = df.groupby(['field','season','dbName']).apply(lambda x: analyze_UD(x),
                                         include_groups=False).reset_index()
    
    return res

def analyze_UD(grp):
    """
    Function to analyze UD seasons

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    res : pandas df
        Result of the analysis.

    """
    
    #grab infos
    dtot = get_infos(grp)
            
    list_var = dtot.keys()
    suffix = '_ud'
    list_var = list(map(lambda x: x+'{}'.format(suffix),list_var))
    
    list_var += ['deltat_beg_survey','deltat_end_survey']
    #analyze UD seasons
    dtot_ud = analyze_season_UD(grp,list_var=list_var,suffix=suffix)
    
    dtot.update(dtot_ud)
    
    print(dtot)
    
    nn = {}
    for key, vals in dtot.items():
        nn[key] = [vals]
        
    res = pd.DataFrame.from_dict(nn)
    
    return res
    
def get_infos(grp,suffix=''):
    """
    Function to grab infos on grp

    Parameters
    ----------
    grp : pandas df
        Data to process.
    suffix : str, optional
        suffix for var name. The default is ''.

    Returns
    -------
    dtot : dict
        result of the analysis.

    """
    
    dtot = {}
    
    #analyze the season
    dtot['season_length{}'.format(suffix)] = season_length(grp)
    
    bands = 'ugrizy'
    for vv in ['nvisits']+list(bands):
        do = get_stat_nvisits(grp,vv,suffix=suffix)
        dtot.update(do)
        
    return dtot
    
    
def analyze_season_UD(grp,list_var,suffix='_ud'):
    """
    Function to analyze a UD season

    Parameters
    ----------
    grp : pandas df
        Data to process.
    list_var : list(str)
        list of variables to fill the dict.
    suffix : str, optional
        suffix for var names. The default is '_ud'.

    Returns
    -------
    ddict : dict
        result of the analysis.

    """
    
    nvisits_tot = grp['nvisits'].sum()
    
    ddict = {}
    if nvisits_tot < 3000:
        ddict = dict(zip(list_var,[-1]*len(list_var)))
        
    else:
    
        #grab the season length in the UD mode
        idx = grp['nvisits'] > 50
        sel = grp[idx]
        ddict = get_infos(sel,suffix=suffix)
    
        deltat_a = get_deltat(grp, sel)
        deltat_b = get_deltat(grp, sel,op=np.max)
        ddict['deltat_beg_survey'] = deltat_a
        ddict['deltat_end_survey'] = deltat_b
    
    return ddict
    
def season_length(df,col='night'):
    """
    Function to estimate the season length

    Parameters
    ----------
    df : pandas df
        Data to process.
    col : str, optional
        column of interest. The default is 'night'.

    Returns
    -------
    season_length : float
        season length.

    """
    
    min_season = df[col].min()
    max_season = df[col].max()
    season_length = max_season-min_season
    
    return int(season_length)
    
def get_deltat(dfa,dfb,col='night',op=np.min):
    """
    Function to estimate delta times

    Parameters
    ----------
    dfa : pandas df
        first dataset to process.
    dfb : pandas df
        second dataset to process.
    col : str, optional
        column to consider. The default is 'night'.
    op : operator, optional
        operator to apply. The default is np.min.

    Returns
    -------
    delta_t : float
        delta_time.

    """
    
    min_seasona = op(dfa[col])
    min_seasonb = op(dfb[col])
    
    delta_t = min_seasonb-min_seasona
    
    return delta_t
    
def get_stat_nvisits(grp,col='nvisits',suffix=''):
    """
    Funtion to grab stat on nvisits

    Parameters
    ----------
    grp : pandas df
        Data to process.
    col : str, optional
        column of interest. The default is 'nvisits'.
    suffix : str, optional
        suffix to add to col. The default is ''.

    Returns
    -------
    dict_df : dict
        output result.

    """
    
    idx = grp[col] > 0
    sel = grp[idx]
    
    dict_df = {}
    dict_df['{}_med{}'.format(col,suffix)] = np.round(sel[col].median(),1)
    dict_df['{}_mean{}'.format(col,suffix)] = np.round(sel[col].mean(),1)
    dict_df['{}_std{}'.format(col,suffix)]= np.round(sel[col].std(),2)
    
    return dict_df
    
parser = OptionParser(
    description='Script to analyse DDF visits on a nightly basis from pointings')

parser.add_option("--dbDir", type="str",
                  default='../ddf_visits_night',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='baseline_v5.3.0_10yrs',
                  help="OS to process [%default]")
parser.add_option("--outDir", type="str",
                  default='../ddf_visits_night_ud',
                  help="output directory [%default]")


opts, args = parser.parse_args()

# Load parameters
dbDir = opts.dbDir
dbName = opts.dbName
outDir = '{}/{}'.format(opts.outDir,dbName)

#create output dir
checkDir(outDir)

#load the data
data = load_data(dbDir, [dbName])

#analyze the data
res = analyze_DDF(data)

#save the data
outName = '{}/ddf_visits_season_ud.hdf5'.format(outDir)

res.to_hdf(outName,key='ddf_ud')


