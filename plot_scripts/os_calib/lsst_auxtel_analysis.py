#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:55:33 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
from sn_plotter_analysis import plt
from sn_plotter_scheduler.plot_scheduler import LSST_plus_AuxTel_visu

def load_data(params):
    
    auxtypes = params['auxtel_type'].split(',')
    
    df = {}
    for auxtype in auxtypes:
        fName = '{}/lsst_plus_auxtel_{}_{}_{}.hdf5'.format(params['dataDir'],
                                                           params['dbName'],
                                                           auxtype,
                                                           params['year'])
    
        dfa = pd.read_hdf(fName)
        dfa['delta_mjd [min]'] = dfa['delta_mjd']*60.*24.
        df[auxtype] = dfa
    
    return df
    
def load_auxtel_data(params):
    
    auxtypes = params['auxtel_type'].split(',')
    
    df_aux = {}
    for auxtype in auxtypes:
        auxName = '{}/auxtel_survey_{}_{}_{}.hdf5'.format(params['auxtel_dir'],
                                                          params['dbName'],
                                                          auxtype,
                                                          params['year'])
        df_aux[auxtype] = pd.read_hdf(auxName)
    
    return df_aux
    
    

def plot_global(df,xvar='delta_mjd [min]',xlabel='$\Delta T$ [min]'):
    
    fig, ax = plt.subplots(figsize=(12,8))

    for key,dfb in df.items():
        ax.hist(dfb[xvar],histtype='step',bins=100)
        #ax.hist(df['delta_r'],histtype='step',bins=50)
    
    ax.grid(visible=True)
    ax.set_xlabel(r'{}'.format(xlabel))
    ax.set_ylabel(r'Number of Entries')
    

parser = OptionParser(description='lsst matched auxtel analysis')

confDict = make_dict_from_config('input_script', 'lsst_auxtel_analysis.txt')

# parser for simulation parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

params=vars(opts)

#load data
df = load_data(params)

"""
plot_global(df)
plot_global(df,xvar='delta_r',xlabel='dist [deg]')
"""

#load auxtel obs
df_aux = load_auxtel_data(params)

aux_type_choice='synchronous'
dd = df[aux_type_choice]
dd_aux = df_aux[aux_type_choice]
dd_aux['year'] = dd_aux['night']/365+1
dd_aux['year'] = dd_aux['year'].astype(int)
print(dd.columns)
visu = LSST_plus_AuxTel_visu(dd.to_records(index=False),
                             dd_aux.to_records(index=False),
                             params['dbName'],aux_type_choice,
                             nside=16,
                             figDir=params['figDir'],
                             movieDir=params['movieDir'],
                             year=params['year'])

nights = dd['night'].unique()
visu(nights,nproc=params['nproc'])


plt.show()