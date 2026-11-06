#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:12:14 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
from sn_analysis.sn_os_atmos_calib_tools import load_OS,ddf_wfd
import numpy as np
import pandas as pd
from sn_tools.sn_utils import multiproc
from sn_tools.os_auxtel_tools import dist
from sn_tools.sn_io import checkDir 

def load_all_obs(params):
    """
    Function to load lsst obs

    Parameters
    ----------
    params : dict
        parameters.

    Returns
    -------
    obs : array
        observations.

    """
    
    #load obs

    obs = load_OS(params['dbDir'], params['dbName'], params['dbExtens'])
    
    #add year
    
    obs['year'] = obs['night']/366+1
    
    obs['year'] = obs['year'].astype(int)
    
    nside=16
    ddf,wfd = ddf_wfd(obs.to_records(index=False),nside=nside)
    
    #concatenate
    
    obs = np.concat((ddf,wfd))
    
    obs.sort(order='mjd')
    
    print(obs.dtype)
    
    idx = obs['year'] == params['year']
    
    obs = obs[idx]
    return obs

def match_lsst_auxtel(lsst_obs,auxtel_obs,nproc=1):
    """
    Main function to match LSST to AuxTel obs

    Parameters
    ----------
    lsst_obs : pandas df
        LSST obs.
    auxtel_obs : pandas df
        AuxTel obs.
    nproc : int, optional
        number of proc for multiprocessing. The default is 1.

    Returns
    -------
    df : pandas df
        output: lsst obs matched to AuxTel obs.

    """
    
    nights = lsst_obs['night'].unique()
    
    params = {}
    
    params['lsst_obs'] = lsst_obs
    params['auxtel_obs'] = auxtel_obs
    
    if nproc == 1:
        df = match_nights(nights,params)
    else:
        df = multiproc(nights,params,match_nights,nproc)
        
    return df
    
def match_nights(nights,params,j=0,output_q=None):
    """
    match LSST/Auxtel for nights (multiprocessing)

    Parameters
    ----------
    nights : list(int)
        List of the nights to process.
    params : dict
        parameters.
    j : int, optional
        internal int for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        where to store output data. The default is None.

    Returns
    -------
    pandas df
        output data.

    """
    
    lsst_obs = params['lsst_obs']
    auxtel_obs = params['auxtel_obs']
    
    #select obs corresponding to this set of nights
    idx = lsst_obs['night'].isin(nights)
    lsst_obs = lsst_obs[idx]
    
    idx = auxtel_obs['night'].isin(nights)
    auxtel_obs = auxtel_obs[idx]
    
    cols = ['observationId','night','year']
    df = lsst_obs.groupby(cols).apply(lambda x: match_night(x,auxtel_obs),
                                        include_groups=False).reset_index()
    
    if output_q is not None:
         return output_q.put({j: df})
    else:
         return df  
     
def match_night(obs,auxtel_obs):
    """
    Single night matching

    Parameters
    ----------
    obs : pandas df
        LSST obs.
    auxtel_obs : pandas df
        auxtel obs.

    Returns
    -------
    pandas df
        matched LSST/Auxtel obs.

    """

    
    #print(obs.name)
    
    night = obs.name[1]
    idx = auxtel_obs['night'] == night
    
    cols_aux = ['Star_name','RA_deg','Dec_deg','mjd_star']
    sel_aux = auxtel_obs[idx][cols_aux]
    
    cols_obs = ['RA','Dec','mjd','visitTime','slewTime']
    cols_obs += ['healpixID', 'pixRA', 'pixDec', 'ebvofMW', 'field']

    sel_obs = obs[cols_obs]
    
    #match
    
    df = sel_obs.merge(sel_aux,how='cross')
    
    df['delta_mjd'] = np.abs(df['mjd']-df['mjd_star'])
    df['delta_r'] = dist(df['RA'],df['Dec'],df['RA_deg'],df['Dec_deg'])
    df['dist_tot'] = np.sqrt((df['delta_mjd']*24*60)**2+df['delta_r']**2)
    #print(df)
    
    df = df.sort_values(by=['dist_tot'])
    
    return df.iloc[0]

parser = OptionParser(description='auxtel survey design')

confDict = make_dict_from_config('input_script', 'match_lsst_auxtel.txt')

# parser for simulation parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

params=vars(opts)

outDir = params['outDir']
#prepare output
if  outDir != 'None':
    checkDir(outDir)

#load LSST survey
lsst_obs = pd.DataFrame.from_records(load_all_obs(params))

print('nobs',len(lsst_obs))

#load AuxTel survey
auxtel_fName = '{}_{}_{}_{}.hdf5'.format(params['auxtel_survey_prefix'],
                                         params['auxtel_survey_db'],
                                         params['auxtel_survey_type'],
                                         params['year'],)
auxtel_obs = pd.read_hdf('{}/{}'.format(params['auxtel_survey_dir'],
                                        auxtel_fName))


auxtel_obs = auxtel_obs.rename(columns={'mjd':'mjd_star'})
print(auxtel_obs)

res = match_lsst_auxtel(lsst_obs, auxtel_obs,nproc=params['nproc'])

print(res)
if outDir != 'None':
    outName = 'lsst_plus_auxtel_{}_{}_{}.hdf5'.format(params['dbName'],
                                                         params['auxtel_survey_type'],
                                                         params['year'])
    res.to_hdf('{}/{}'.format(outDir,outName),key='lsst_plus_auxtel')