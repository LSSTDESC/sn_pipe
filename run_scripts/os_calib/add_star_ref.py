#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:31:15 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
#import pandas as pd
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
import numpy as np
from sn_tools.sn_utils import multiproc
import time
from sn_tools.sn_io import checkDir
from sn_analysis.sn_os_atmos_calib_tools import load_OS,load_star_calib

def match_obs_calib_stars_multiproc(obs, star_calib):
    """
    match obs/star for calib using mumltiprocessing

    Parameters
    ----------
    obs : pandas df
        Data to process.
    star_calib : pandas df
        list of stars to match obs..

    Returns
    -------
    ttf : pandas df
        original df plus star_name,ra,dec,dist.

    """
    
    params = {}
    
    params['obs'] = obs
    params['star_calib'] = star_calib
    
    col = 'observationId'
    params['col'] = col
    
    ll = obs[col].to_list()
    
    ttf = multiproc(ll,params,match_obs_calib_stars,nproc=24)
    
    return ttf
    
def match_obs_calib_stars(toproc, params, j=0, output_q=None):
    """
    
    match obs/star for calib for a set of obs.
    
    Parameters
    ----------
    toproc : list(str)
        list to process.
    params : dict
        parameters.
    j : int, optional
        internal int for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        where to store the data. The default is None.

    Returns
    -------
    pandas df
        output data.

    """
    
    obs = params['obs']
    star_calib = params['star_calib']
    col = params['col']
    
    idx = obs[col].isin(toproc)
    
    obsb = obs[idx]
    
    tt = obsb.groupby([col]).apply(lambda x:matchIt(x,star_calib),include_groups=False).reset_index()
    
    if output_q is not None:
        return output_q.put({j:tt})
    else:
        return tt

def matchIt(grp,stars):
    """
    Function to grab the nearest calib star to an obs 

    Parameters
    ----------
    grp : pandas df
        observations.
    stars : pandas df
        calib stars.

    Returns
    -------
    pandas df
        original df + nearest star.

    """
    
    cols = ['Star_name','RA_deg','Dec_deg']
    dfb = grp.merge(stars[cols],how='cross',suffixes=['',''])
    
    diff_ra = ((dfb['RA']-dfb['RA_deg'])*np.cos(np.deg2rad(dfb['Dec'])))**2
    diff_dec = (dfb['Dec']-dfb['Dec_deg'])**2
    dfb['dist [deg]'] = np.sqrt(diff_ra+diff_dec)
    dfb = dfb.sort_values(by=['dist [deg]'])
    
    return dfb.iloc[0]

parser = OptionParser(description='add (smeared) atmos. params to OS strategy')

confDict = make_dict_from_config('input_script', 'add_star_ref.txt')

# parser for simulation parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

params=vars(opts)

#load star calib
star_calib = load_star_calib(params['star_calib_file'])

#load obs

obs = load_OS(params['dbDir'], params['dbName'], params['dbExtens'])

#match obs/star_calib
time_ref = time.time()
tt = match_obs_calib_stars_multiproc(obs, star_calib)

## save the data
checkDir(params['outDir'])

fName = '{}/{}_star_calib.npy'.format(params['outDir'],params['dbName'])

res = tt.to_records(index=False)

np.save(fName,res)
print(time.time()-time_ref)

