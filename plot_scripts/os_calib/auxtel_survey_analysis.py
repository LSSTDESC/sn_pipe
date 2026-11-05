#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:27:07 2026

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
from sn_plotter_analysis import plt
from sn_plotter_tools.plot_tools import plot_indiv
import numpy as np
np.set_printoptions(legacy='1.25')

def ana_data(dd):
    
    dft = pd.DataFrame()
    for key,df in dd.items():
        df_o = df.groupby(['night']).apply(lambda x:night_analysis(x),
                                           include_groups=False).reset_index()
        df_o['config'] = key
        dft = pd.concat((dft,df_o))
        
    return dft

def night_analysis(df):
    
    dd = {}
    
    dd['nvisits'] = [len(df)]
    dd['slew_time'] = [df['slew_time'].sum()]
    dd['nstars'] = [len(df['Star_name'].unique())]
    df['visitTime'] = df['numExposure']*df['exposure_time']
    df['tot_time'] = df['slew_time']+df['visitTime']
    dd['exposure_time'] = [df['visitTime'].sum()]
    dd['slew_rat'] = [df['slew_time'].sum()/df['tot_time'].sum()]
    dd['total_night_time'] = [df['tot_time'].sum()/3600.]
    
    res = pd.DataFrame.from_dict(dd)
    
    return res
    
def plot_night(df,yvar='nstars',ylabel='$N_{stars}$'):
    
    configs = df['config'].unique()
    
    colors = ['k','r']
    
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    
    for i,conf in enumerate(configs):
        idx = df['config'] == conf
        sel = df[idx]
        plot_indiv(sel,
                   xvar='night',xlabel='night',
                   yvar=yvar,ylabel=ylabel,
                   color=colors[i],
                   fig=fig,ax=ax,label=conf)    
    
    ax.grid(visible=True)
    ax.set_xlabel(r'night')
    ax.set_ylabel(r'{}'.format(ylabel))
    ax.legend()
    
dataDir = '../auxtel_surveys'

files = ['auxtel_survey_baseline_v5.3.0_10yrs_star_calib_asynchronous_1.hdf5',
         'auxtel_survey_baseline_v5.3.0_10yrs_star_calib_synchronous_1.hdf5']
config=['async','sync']

dd = dict(zip(config,files))

df_dict = {}

for key,vals in dd.items():
    fName = '{}/{}'.format(dataDir,vals)
    df_dict[key] = pd.read_hdf(fName)
    

df = ana_data(df_dict)

plot_night(df,yvar='slew_time',ylabel='slew time [s]')
plot_night(df,yvar='slew_rat',ylabel='slew time/total night time')   
plot_night(df,yvar='total_night_time',ylabel='total night time [h]')
plt.show()
