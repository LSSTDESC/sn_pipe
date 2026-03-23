#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:29:11 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_tools.sn_cosmo_model import cosmo_values
import copy
import numpy as np
import pandas as pd

def stat(grp):
    """
    Function to estimate some stat

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    res : pandas df
        results.

    """
    
    dout = {}
    
    for vv in ['mu','dl','w']:
        dout[vv] = [grp[vv].mean()]
        dout['{}_std'.format(vv)] = [grp[vv].std]
    
    res = pd.DataFrame.from_dict(dout)
    
    return res
    

parser = OptionParser(description='Script to analyze SN selection criteria')

parser.add_option('--de_params', type=str,
                  default='w0,wa',
                  help='DE eos parameters [%default]')
parser.add_option('--de_values', type=str,
                  default='-1.,0.',
                  help='DE eos parameter values [%default]')
parser.add_option('--de_sigma_values', type=str,
                  default='0.1,0.1',
                  help='DE eos parameter sigma values [%default]')
parser.add_option('--de_class', type=str,
                  default='w0waCDM',
                  help='DE class to use (w0waCDM/DDE_FLRW) [%default]')
parser.add_option('--class_loc', type=str,
                  default='astropy.cosmology',
                  help='DE class location \
                        (astropy.cosmology/sn_tools.sn_cosmo_model) [%default]')
parser.add_option('--de_model', type=str,
                  default='CPL',
                  help='DE eos model name [%default]')
parser.add_option('--de_eos', type=str,
                  default='w0+wa*z/(1+z)',
                  help='DE eos model [%default]')
parser.add_option('--H0', type=float,
                  default=70.,
                  help='DE eos model [%default]')
parser.add_option('--Om0', type=float,
                  default=0.30,
                  help='Omega_matter [%default]')
parser.add_option('--ntrials', type=int,
                  default=1000,
                  help='n random trials to estimate cosmo [%default]')
parser.add_option('--outName', type='str',
                  default='cosmo_simu',
                  help='output file name [%default]')

opts, args = parser.parse_args()

opt_dict = vars(opts)

de_values = opt_dict['de_values'].split(',')
de_sigma_values = opt_dict['de_sigma_values'].split(',')
de_params = opt_dict['de_params'].split(',')

de_values = list(map(float,de_values))
de_sigma_values = list(map(float,de_sigma_values))
de_sigmas = dict(zip(de_params,de_sigma_values))
ntrials = opt_dict['ntrials']

params = copy.deepcopy(opt_dict)

del params['de_values']
del params['de_sigma_values']
del params['de_params']
del params['ntrials']
del params['outName']
params['de_params'] = dict(zip(de_params,de_values))
params['Ode0'] = 1.-params['Om0']


#grab a list of de parameters values 

dict_params = {}

for key, vals in params['de_params'].items():
    sigma = de_sigmas[key]
    vv = np.random.normal(vals,sigma,ntrials)
    dict_params[key] = np.round(vv,3)
    
df_params = pd.DataFrame.from_dict(dict_params)
#print(df_params)
    
de_p = df_params.columns
dft = pd.DataFrame()
z = np.arange(0.01,1.11,0.01)
for i,row in df_params.iterrows():
    de_val = row[de_params].to_list()
    params['de_params'] = dict(zip(de_p,de_val))
    df = cosmo_values(params,z=z)
    dft = pd.concat((df,dft))

#print(dft.columns)

res = dft.groupby('z').apply(lambda x:stat(x),include_groups=False).reset_index()

ccols = ['z','de_params','de_eos','de_class','class_loc', 'de_model']

tt = dft[ccols].drop_duplicates()
res = res.merge(tt,left_on=['z'],right_on=['z'])
res['de_values'] = opt_dict['de_values']
res['de_sigma_values'] = opt_dict['de_sigma_values']
fName = '{}.hdf5'.format(opt_dict['outName'])

res.to_hdf(fName,key='cosmology')
