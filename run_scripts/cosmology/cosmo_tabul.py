#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:02:31 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import numpy as np
import os
from sn_cosmology.cosmo_tools import load_cosmo_params_from_script
from sn_tools.sn_cosmo_model import cosmo_wrapper
from sn_tools.sn_utils import multiproc
from sn_tools.sn_io import checkDir

def dist_modulus(cosmo_params,z=np.arange(0.01,1.11,0.01)):
    """
    Function to estimate the distance moduli from cosmo estimation

    Parameters
    ----------
    cosmo_params : dict
        cosmo parameters.
    z : array, optional
        List of redshifts. The default is np.arange(0.01,1.11,0.01).

    Returns
    -------
    res : pandas df
        distmod vs z.

    """
    cosmo = cosmo_wrapper(cosmo_params)
        
    distmod = cosmo.distmod(z).value
    
    res = pd.DataFrame(z,columns=['z'])
    
    res['distmod'] = distmod
    
    return res
    

def get_par_values(col,xmin,xmax,delta):
    """
    Function to estimate parameter values

    Parameters
    ----------
    col : str
        column name.
    xmin : float
        min value.
    xmax : float
        max value.
    delta : float
        step value.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    
    vv = np.arange(xmin,xmax+delta,delta)
    
    res = pd.DataFrame(vv,columns=[col])
    
    return res

def tabul_multi(toproc, pp, j=0, output_q=None):
    """
    Function to estimate distance moduli 

    Parameters
    ----------
    toproc : list(int)
        index to process.
    pp : dict
        parameter dict.
    j : int, optional
        internal tag for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        where to copy the results. The default is None.

    Returns
    -------
    pandas df
        Output data.

    """
    
    df_params = pp['data']
    params = pp['cosmodict']
    par_name = pp['par_name']
    
    idx = df_params['num'].isin(toproc)
    
    df_params = df_params[idx]
    
    df_tot = pd.DataFrame()
    for i, row in df_params.iterrows():
        de_values = '{},{}'.format(row[par_name[0]],row[par_name[1]])
        params['devalues'] = de_values
        cosmo_params = load_cosmo_params_from_script(params)
        vv = dist_modulus(cosmo_params)
        for i in range(len(par_name)):
            vv[par_name[i]] = row[par_name[i]]
        df_tot = pd.concat((df_tot,vv))
    
    if output_q is not None:
            return output_q.put({j: df_tot})
    else:
            return df_tot

def build_sample(opts,outName):
    """
    Function to build the sample

    Parameters
    ----------
    opts : opts parser
        parameters of the script.
    outName : str
        output file name.

    Returns
    -------
    None.

    """
    
    
    params = vars(opts)
    
    par_name = opts.deparams.split(',')
    par_min = opts.deparams_min.split(',')
    par_max = opts.deparams_max.split(',')
    par_delta = opts.deparams_delta.split(',')
    
    par_min = list(map(float,par_min))
    par_max = list(map(float,par_max))
    par_delta = list(map(float,par_delta))
    
    df_params = pd.DataFrame()
    
    for i,vv in enumerate(par_name):
        df_ = get_par_values(vv,par_min[i],par_max[i],par_delta[i])
        
        if len(df_params) == 0:
            df_params = df_
        else:
            df_params = df_params.merge(df_,how='cross')
    
    df_params['num'] = df_params.index
    
    print(df_params)
    
    pp = {}
    
    pp['data'] = df_params
    pp['cosmodict'] = params
    pp['par_name'] = par_name
    
    toproc = df_params['num'].to_list()
    
    df_tot = multiproc(toproc,pp,tabul_multi,nproc=8)    
    
    #outName = '{}_{}.hdf5'.format(params['outName'],params['demodel'])
    
    df_tot.to_hdf(outName,key='distmod')
    
def check_interp(opts,df_tot):
    """
    Function to check interpolator

    Parameters
    ----------
    opts : parser opts
        script parameters.
    df_tot : pandas df
        tabulated data.

    Returns
    -------
    None.

    """
    
    from scipy.interpolate import LinearNDInterpolator
       
    par_name = opts.deparams.split(',')
    ccols = []
    for po in par_name:
        ccols.append(po)

    ccols.append('z')
    
    vals = df_tot['distmod']
    interp = LinearNDInterpolator(df_tot[ccols],vals,fill_value=0.) 
        
    dmin={}
    dmax={}
    for vv in ccols:
        dmin[vv] = df_tot[vv].min()
        dmax[vv] = df_tot[vv].max()
    
    nrand = 5

    to = pd.DataFrame()
    for vv in ccols:
        rrand= np.random.uniform(dmin[vv],dmax[vv],nrand)
        if len(to) == 0:
            to = pd.DataFrame(rrand.tolist(),columns=[vv])
        else:
            to[vv] = rrand.tolist()
    
    print('random check')
    print(to)
    
    res = list(interp(to[ccols]))
    
    pp = vars(opts)
    real_val = []
       
    for i,row in to.iterrows():
        rb  =[]
        for j in range(len(par_name)):
            rb.append(row[par_name[j]])
        rb = list(map(str,rb))
        rb = ','.join(rb)
        pp['devalues'] = rb
        cosmo_params = load_cosmo_params_from_script(pp)
        estim_val =  dist_modulus(cosmo_params,[row['z']])['distmod'].values[0]
        real_val.append(estim_val)
        
   
    for i in range(len(res)):
        print(i,res[i],real_val[i],res[i]/real_val[i])
        
parser = OptionParser(description='Script to estimate distmod in 3D')

parser.add_option('--deparams', type=str,
                  default='w1,w2',
                  help='DE eos parameters [%default]')
parser.add_option('--deparams_min', type=str,
                  default='-0.5,-3.',
                  help='DE eos parameter min values [%default]')
parser.add_option('--deparams_max', type=str,
                  default='0.,-1.',
                  help='DE eos parameter max values [%default]')
parser.add_option('--deparams_delta', type=str,
                  default='0.01,0.01',
                  help='DE eos parameter n values [%default]')
parser.add_option('--declass', type=str,
                  default='DDE_FLRW',
                  help='DE class to use (w0waCDM/DDE_FLRW) [%default]')
parser.add_option('--classloc', type=str,
                  default='sn_tools.sn_cosmo_model',
                  help='DE class location \
                        (astropy.cosmology/sn_tools.sn_cosmo_model) [%default]')
parser.add_option('--demodel', type=str,
                  default='oscilla',
                  help='DE eos model name [%default]')
parser.add_option('--deeos', type=str,
                  default='-1.+(w1*z*np.sin(w2*z))/(1+z**2)',
                  help='DE eos model [%default]')
parser.add_option('--H0', type=float,
                  default=70.,
                  help='DE eos model [%default]')
parser.add_option('--Om0', type=float,
                  default=0.30,
                  help='Omega_matter [%default]')
parser.add_option('--Ode0', type=float,
                  default=0.70,
                  help='Omega_DE [%default]')
parser.add_option('--outName', type=str,
                  default='distmod_tabul',
                  help='prefix for output name [%default]')
parser.add_option('--outDir', type=str,
                  default='../distmod_tabul',
                  help='output directory [%default]')

opts, args = parser.parse_args()

checkDir(opts.outDir)
outName = '{}/{}_{}.hdf5'.format(opts.outDir,opts.outName,opts.demodel)

if not os.path.isfile(outName):
    build_sample(opts,outName)
    
df_tot = pd.read_hdf(outName)
    
check_interp(opts,df_tot)


